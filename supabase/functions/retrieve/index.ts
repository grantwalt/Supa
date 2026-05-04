import express from "express";
import path from "path";
import dotenv from "dotenv";
import { readFileSync } from "fs";
import { GoogleGenAI } from "@google/genai";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

// --- Configuration Constants ---
const SIMILARITY_THRESHOLD = 0.75;
const CONFIDENCE_THRESHOLD = 0.75;
const MAX_REQUESTS_PER_SESSION = 60;
const MIN_QUESTION_LENGTH = 8;
const RATE_WINDOW_SECONDS = 3600;

// --- Supabase Client (Service Role for Backend) ---
const supabaseUrl = process.env.SUPABASE_URL || "";
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const supabase = createClient(supabaseUrl, supabaseKey);

// --- Google Gen AI (Vertex AI mode) Setup ---
let genAIInstance: GoogleGenAI | null = null;
const getVertexAI = () => {
  if (!genAIInstance) {
    let project = process.env.VERTEX_PROJECT_ID || process.env.GOOGLE_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT;
    const location = process.env.VERTEX_LOCATION || "us-central1";

    if (!project || project.startsWith("gen-lang-client")) {
      try {
        const configPath = path.join(process.cwd(), "firebase-applet-config.json");
        const config = JSON.parse(readFileSync(configPath, "utf-8"));
        if (config.projectId && !config.projectId.startsWith("gen-lang-client")) {
          project = config.projectId;
        }
      } catch (e) { /* ignore */ }
    }

    if (!project || project.startsWith("gen-lang-client")) {
      throw new Error(`Invalid Project ID: "${project || 'MISSING'}". Standard GCP projects are required for Vertex AI.`);
    }

    const googleCredentials = process.env.GOOGLE_CREDENTIALS || process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
    let googleAuthOptions = undefined;
    if (googleCredentials) {
      try {
        // Support both plain JSON and base64-encoded JSON
        let credJson = googleCredentials;
        if (!googleCredentials.trim().startsWith("{")) {
          credJson = Buffer.from(googleCredentials, "base64").toString("utf-8");
        }
        googleAuthOptions = { credentials: JSON.parse(credJson) };
      } catch (e) {
        console.warn("API: Credentials JSON is not valid.");
      }
    }

    genAIInstance = new GoogleGenAI({ 
      vertexai: true,
      project, 
      location, 
      googleAuthOptions 
    });
  }
  return genAIInstance;
};

// --- Helper Functions ---

async function checkRateLimit(sessionId: string) {
  const { data: session } = await supabase
    .from("sessions")
    .select("request_count, last_request_at")
    .eq("session_id", sessionId)
    .maybeSingle();

  const now = Date.now();
  const windowMs = RATE_WINDOW_SECONDS * 1000;
  const lastRequest = session ? new Date(session.last_request_at).getTime() : 0;
  const resetCount = !session || (now - lastRequest > windowMs);
  const newCount = resetCount ? 1 : session.request_count + 1;

  if (newCount > MAX_REQUESTS_PER_SESSION) return { allowed: false };

  await supabase.from("sessions").upsert({
    session_id: sessionId,
    request_count: newCount,
    last_request_at: new Date().toISOString(),
  }, { onConflict: "session_id" });

  return { allowed: true };
}

async function fetchByIntent(caseId: string, intentId: string, personality: string) {
  const { data: exact } = await supabase
    .from("response_cache")
    .select("*")
    .eq("case_id", caseId)
    .eq("intent_id", intentId)
    .eq("personality", personality)
    .eq("reviewed", true)
    .limit(1)
    .maybeSingle();
  
  if (exact) return exact;

  const { data: neutral } = await supabase
    .from("response_cache")
    .select("*")
    .eq("case_id", caseId)
    .eq("intent_id", intentId)
    .eq("personality", "neutral")
    .eq("reviewed", true)
    .limit(1)
    .maybeSingle();
  
  return neutral;
}

async function embedText(text: string) {
  const ai = getVertexAI();
  const embeddingModel = ai.models.get({ model: "text-embedding-004" });
  const result = await embeddingModel.embedContent({
    content: { role: "user", parts: [{ text }] },
    taskType: "RETRIEVAL_QUERY"
  });
  return result.embedding.values;
}

const app = express();
app.use(express.json());

app.get("/api/health", async (req, res) => {
  const vId = process.env.VERTEX_PROJECT_ID;
  res.json({ 
    status: "ok", 
    supabase: !!supabaseUrl,
    vertex: vId || "MISSING"
  });
});

app.post("/api/ai/action", async (req, res) => {
  try {
    const { case_id, question, intent_id, personality, session_id, mode } = req.body;
    const resolvedPersonality = personality || "neutral";

    if (session_id) {
      const rate = await checkRateLimit(session_id);
      if (!rate.allowed) return res.status(429).json({ error: "Rate limit exceeded" });
    }

    if (intent_id) {
      const match = await fetchByIntent(case_id, intent_id, resolvedPersonality);
      if (match) {
        return res.json({
          reply: match.response_text,
          intent_id: match.intent_id,
          response_type: match.response_type,
          personality: match.personality,
          source: "cache_exact",
          reviewed: match.reviewed
        });
      }
    }

    let embedding = null;
    if (mode === "ai" && question.length >= MIN_QUESTION_LENGTH) {
      try {
        embedding = await embedText(question);
        const { data: vectorMatch } = await supabase.rpc("match_response_cache", {
          query_embedding: embedding,
          match_case_id: case_id,
          match_personality: resolvedPersonality,
          similarity_threshold: SIMILARITY_THRESHOLD,
          match_count: 1
        });
        
        if (vectorMatch && vectorMatch.length > 0) {
          const match = vectorMatch[0];
          return res.json({
            reply: match.response_text,
            intent_id: match.intent_id,
            response_type: match.response_type,
            personality: match.personality,
            source: "cache_vector",
            reviewed: match.reviewed
          });
        }
      } catch (e) { /* silent fail on vector search for UX stability */ }
    }

    const ai = getVertexAI();

    // Fetch full case data so the LLM has patient profile context
    const { data: caseData } = await supabase
      .from("cases")
      .select("*")
      .eq("case_id", case_id)
      .maybeSingle();

    const patientProfile = caseData?.patient_profile ?? {};
    const caseContext    = caseData?.case_context ?? "";

    const systemInstruction = `
You are roleplaying as a patient in a Nigerian teaching hospital OSCE examination.
Patient profile: ${JSON.stringify(patientProfile)}
Case context: ${caseContext}
Personality: ${resolvedPersonality}

Respond ONLY with a valid JSON object. No markdown, no backticks, no preamble.
Schema:
{
  "reply":     "your response as the patient (10–400 characters)",
  "intent":    "snake_case intent id if you can infer it, else null",
  "category":  "history" | "exam" | "investigation" | "social"
}

Rules:
- Never reveal the diagnosis directly.
- Never break character or mention AI, scenarios, or training.
- Answer must be consistent with the patient profile (age, gender, presenting complaint).
- If irrelevant, give a brief dismissive reply staying in character.
    `.trim();

    const result = await ai.models.generateContent({
      model: "gemini-2.5-flash-lite",
      config: {
        systemInstruction: { role: "system", parts: [{ text: systemInstruction }] },
      },
      contents: [{ role: "user", parts: [{ text: question }] }]
    });

    const text = result.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
    const parsed = JSON.parse(text);

    if (supabaseUrl) {
      (async () => {
        try {
          await supabase.from("unmatched_log").insert({
            case_id,
            raw_question: question,
            llm_reply: parsed.reply,
            intent_id: parsed.intent || intent_id,
            category: parsed.category,
            embedding: embedding,
            session_id
          });
        } catch (e) {}
      })();
    }

    res.json({
      reply: parsed.reply,
      intent_id: parsed.intent || intent_id,
      response_type: parsed.category || "history",
      personality: resolvedPersonality,
      source: "llm_live",
      reviewed: false
    });

  } catch (error: any) {
    console.error("AI Error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/cases", async (req, res) => {
  try {
    const { data: cases, error } = await supabase
      .from("cases")
      .select("*")
      .eq("active", true);
    
    if (error) throw error;
    res.json(cases);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

export default app;