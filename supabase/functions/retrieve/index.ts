// ============================================================
//  CLERKAI — SUPABASE EDGE FUNCTION
//  /functions/v1/retrieve
//
//  Retrieval pipeline:
//    1. Intent lookup (exact match — fast, free)
//    2. Vector similarity search (fallback — costs embedding)
//    3. LLM live call (last resort — costs most)
//       └─ Validates response
//       └─ Saves to unmatched_log or response_cache
//
//  Deploy:
//    supabase functions deploy retrieve
// ============================================================

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// ── CONSTANTS ───────────────────────────────────────────────

const SIMILARITY_THRESHOLD = 0.75;   // minimum cosine score for vector match (was 0.82 — lowered to catch more good matches)
const CONFIDENCE_THRESHOLD = 0.75;   // minimum model_confidence to serve unreviewed entry
const MAX_REQUESTS_PER_SESSION = 60; // rate limit per session
const MAX_LLM_CALLS_PER_SESSION = 10; // hard LLM cap per hour — protects $300 credit
const MIN_QUESTION_LENGTH = 8;       // skip embedding for trivially short inputs
const RATE_WINDOW_SECONDS = 3600;    // 1 hour window

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// ── TYPES ───────────────────────────────────────────────────

interface RetrieveRequest {
  case_id:    string;
  question:   string;
  intent_id?: string | null;   // resolved by client-side intent engine
  personality: string;
  session_id: string;
  mode:       "classic" | "ai";
}

interface CacheEntry {
  id:            string;
  response_text: string;
  response_type: string;
  intent_id:     string;
  personality:   string;
  source:        string;
  reviewed:      boolean;
  hit_count:     number;
}

interface RetrieveResponse {
  reply:        string;
  intent_id:    string | null;
  response_type: string;
  personality:  string;
  source:       string;          // "cache_exact" | "cache_vector" | "llm_live" | "fallback"
  reviewed:     boolean;
  cache_id:     string | null;   // for hit count increment
  score?:       number;          // intent score for frontend scoring engine
}

// ── MAIN HANDLER ────────────────────────────────────────────

serve(async (req: Request) => {

  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS });
  }

  if (req.method !== "POST") {
    return error(405, "Method not allowed");
  }

  // ── Parse & validate request body
  let body: RetrieveRequest;
  try {
    body = await req.json();
  } catch {
    return error(400, "Invalid JSON body");
  }

  const { case_id, question, intent_id, personality, session_id, mode } = body;

  if (!case_id || !question || !session_id || !mode) {
    return error(400, "Missing required fields: case_id, question, session_id, mode");
  }

  // ── Supabase client (service_role — bypasses RLS)
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  );

  // ── Latency tracking
  const t0 = performance.now();
  const latency: Record<string, number> = {};

  // ── Rate limiting
  const rateLimitResult = await checkRateLimit(supabase, session_id);
  if (!rateLimitResult.allowed) {
    return error(429, `Rate limit exceeded. Max ${MAX_REQUESTS_PER_SESSION} requests per hour.`);
  }

  // ── Fetch case context (needed for LLM fallback)
  const { data: caseData, error: caseError } = await supabase
    .from("cases")
    .select("case_id, patient_profile, case_context, intent_map, scoring_map, diagnosis")
    .eq("case_id", case_id)
    .eq("active", true)
    .maybeSingle();

  if (caseError || !caseData) {
    return error(404, `Case not found: ${case_id}`);
  }

  // ── Normalised personality — default neutral if invalid
  const resolvedPersonality = normalisePersonality(personality);

  // ════════════════════════════════════════════════════════
  //  STAGE 1 — EXACT INTENT LOOKUP
  //  Fastest path. Uses intent_id from client intent engine.
  //  Queries response_cache by case_id + intent_id + personality.
  // ════════════════════════════════════════════════════════

  if (intent_id) {
    const t1 = performance.now();
    const exactMatch = await fetchByIntent(
      supabase, case_id, intent_id, resolvedPersonality
    );

    if (exactMatch) {
      latency.stage1_ms = Math.round(performance.now() - t1);
      console.log(`[retrieve] cache_exact | intent=${intent_id} | ${latency.stage1_ms}ms`);
      incrementHitCount(supabase, exactMatch.id);
      trackAnalytics(supabase, { session_id, case_id, stage: "exact", source: "cache_exact", latency_ms: latency.stage1_ms, intent_id });

      return respond({
        reply:         exactMatch.response_text,
        intent_id:     exactMatch.intent_id,
        response_type: exactMatch.response_type,
        personality:   exactMatch.personality,
        source:        "cache_exact",
        reviewed:      exactMatch.reviewed,
        cache_id:      exactMatch.id,
        score:         getIntentScore(caseData, intent_id),
      });
    }
  }

  // ════════════════════════════════════════════════════════
  //  STAGE 2 — CLASSIC MODE HARD STOP (before embedding)
  //  Classic mode falls back to intentMap text from the
  //  case object if Stage 1 missed. Never embeds, never
  //  calls LLM. Moved here to avoid wasting embedding cost.
  // ════════════════════════════════════════════════════════

  if (mode === "classic") {
    const classicReply = getClassicFallback(caseData, intent_id);
    latency.stage2_classic_ms = Math.round(performance.now() - t0);
    console.log(`[retrieve] classic_fallback | intent=${intent_id ?? "none"} | ${latency.stage2_classic_ms}ms`);

    return respond({
      reply:         classicReply.text,
      intent_id:     intent_id ?? null,
      response_type: classicReply.type,
      personality:   resolvedPersonality,
      source:        "fallback",
      reviewed:      true,
      cache_id:      null,
      score:         classicReply.score,
    });
  }

  // ════════════════════════════════════════════════════════
  //  STAGE 3 — VECTOR SIMILARITY SEARCH (AI mode only)
  //  Skip if question too short — not worth embedding cost.
  //  Only serves matches above SIMILARITY_THRESHOLD.
  // ════════════════════════════════════════════════════════

  const t2 = performance.now();
  let questionEmbedding: number[] | null = null;

  if (question.length >= MIN_QUESTION_LENGTH) {
    try {
      questionEmbedding = await embedText(question, supabase);
    } catch (e) {
      console.warn("[retrieve] Embedding failed, skipping vector search:", e);
    }
  } else {
    console.log(`[retrieve] question too short (${question.length} chars) — skipping embedding`);
  }

  if (questionEmbedding && questionEmbedding.length > 0) {
    const vectorMatch = await fetchByVector(
      supabase, case_id, questionEmbedding, resolvedPersonality
    );

    if (vectorMatch) {
      latency.stage3_ms = Math.round(performance.now() - t2);
      console.log(`[retrieve] cache_vector | intent=${vectorMatch.intent_id} | ${latency.stage3_ms}ms`);
      incrementHitCount(supabase, vectorMatch.id);
      trackAnalytics(supabase, { session_id, case_id, stage: "vector", source: "cache_vector", latency_ms: latency.stage3_ms, intent_id: vectorMatch.intent_id });

      return respond({
        reply:         vectorMatch.response_text,
        intent_id:     vectorMatch.intent_id,
        response_type: vectorMatch.response_type,
        personality:   vectorMatch.personality,
        source:        "cache_vector",
        reviewed:      vectorMatch.reviewed,
        cache_id:      vectorMatch.id,
        score:         getIntentScore(caseData, vectorMatch.intent_id),
      });
    }
  }

  // ════════════════════════════════════════════════════════
  //  STAGE 4 — LLM LIVE CALL (AI mode only)
  //  Hard cap: MAX_LLM_CALLS_PER_SESSION per hour.
  //  Last resort. Calls Vertex AI, validates the response,
  //  saves to unmatched_log or response_cache.
  // ════════════════════════════════════════════════════════

  // ── LLM cap guard — prevents runaway cost
  const llmCapResult = await checkLLMCap(supabase, session_id);
  if (!llmCapResult.allowed) {
    const fallback = getClassicFallback(caseData, intent_id, resolvedPersonality);
    console.warn(`[retrieve] LLM cap reached for session ${session_id} — forcing fallback`);
    return respond({
      reply:         fallback.text,
      intent_id:     intent_id ?? null,
      response_type: fallback.type,
      personality:   resolvedPersonality,
      source:        "fallback",
      reviewed:      true,
      cache_id:      null,
    });
  }

  const t4 = performance.now();
  let llmResult: LLMResult | null = null;

  try {
    llmResult = await callLLM(question, caseData, resolvedPersonality);
  } catch (e) {
    console.error("LLM call failed:", e);
  }

  if (!llmResult) {
    // LLM unreachable — use classic fallback for consistent UX
    const fallback = getClassicFallback(caseData, intent_id, resolvedPersonality);
    console.warn("[retrieve] LLM call failed, using classic fallback");
    return respond({
      reply:         fallback.text,
      intent_id:     intent_id ?? null,
      response_type: fallback.type,
      personality:   resolvedPersonality,
      source:        "fallback",
      reviewed:      true,
      cache_id:      null,
    });
  }

  // ── Validate LLM response
  const validation = validateLLMResponse(llmResult, caseData);

  if (!validation.pass) {
    // Log failure, serve safe fallback
    await logError(supabase, {
      case_id,
      raw_question:     question,
      llm_raw_response: JSON.stringify(llmResult),
      failure_rule:     validation.rule!,
      failure_detail:   validation.reason!,
    });

    return respond({
      reply:         getClassicFallback(caseData, intent_id, resolvedPersonality).text,
      intent_id:     intent_id ?? null,
      response_type: "history",
      personality:   resolvedPersonality,
      source:        "fallback",
      reviewed:      false,
      cache_id:      null,
    });
  }

  // ── Save validated response
  const resolvedIntentId = llmResult.intent || intent_id || null;

  // Fix: check intents table (source of truth), not intent_map JSON
  let intentExists = false;
  if (resolvedIntentId) {
    const { data: intentRow } = await supabase
      .from("intents")
      .select("id")
      .eq("id", resolvedIntentId)
      .eq("case_id", case_id)
      .maybeSingle();
    intentExists = !!intentRow;
  }

  let savedId: string | null = null;

  if (intentExists && resolvedIntentId) {
    // Auto-promote if confidence is very high — reduces manual review queue
    const autoReviewed = llmResult.confidence >= 0.90;

    // Upsert prevents duplicate auto-saves for same case+intent+personality
    const { data: saved } = await supabase
      .from("response_cache")
      .upsert({
        case_id,
        intent_id:        resolvedIntentId,
        response_text:    llmResult.answer,
        response_type:    llmResult.category as string,
        personality:      resolvedPersonality,
        source:           "auto",
        reviewed:         autoReviewed,
        model_confidence: llmResult.confidence,
        embedding:        questionEmbedding ?? undefined,
      }, {
        onConflict:        "case_id,intent_id,personality",
        ignoreDuplicates:  false,   // update if exists (keeps best confidence)
      })
      .select("id")
      .maybeSingle();

    if (autoReviewed) {
      console.log(`[retrieve] auto-promoted | intent=${resolvedIntentId} | confidence=${llmResult.confidence}`);
    }

    savedId = saved?.id ?? null;
  } else {
    // Intent unknown → save to unmatched_log for review
    await supabase.from("unmatched_log").insert({
      case_id,
      raw_question:     question,
      cleaned_question: llmResult.cleaned_question,
      llm_reply:        llmResult.answer,
      intent_id:        resolvedIntentId,
      keywords:         llmResult.keywords,
      category:         llmResult.category,
      embedding:        questionEmbedding ?? undefined,   // null guard
      session_id,
      reviewed:         false,
      promoted:         false,
    });
  }

  latency.stage4_ms = Math.round(performance.now() - t4);
  console.log(`[retrieve] llm_live | intent=${resolvedIntentId ?? "unknown"} | ${latency.stage4_ms}ms`);
  trackAnalytics(supabase, { session_id, case_id, stage: "llm", source: "llm_live", latency_ms: latency.stage4_ms, intent_id: resolvedIntentId ?? null });

  return respond({
    reply:         llmResult.answer,
    intent_id:     resolvedIntentId,
    response_type: llmResult.category,
    personality:   resolvedPersonality,
    source:        "llm_live",
    reviewed:      false,
    cache_id:      savedId,
    score:         resolvedIntentId
      ? getIntentScore(caseData, resolvedIntentId)
      : 0,
  });
});


// ── STAGE 1 HELPER — fetch by intent ────────────────────────

async function fetchByIntent(
  supabase:    ReturnType<typeof createClient>,
  case_id:     string,
  intent_id:   string,
  personality: string,
): Promise<CacheEntry | null> {

  // Try exact personality first
  const { data: exact } = await supabase
    .from("response_cache")
    .select("id, response_text, response_type, intent_id, personality, source, reviewed, hit_count")
    .eq("case_id", case_id)
    .eq("intent_id", intent_id)
    .eq("personality", personality)
    .eq("reviewed", true)
    .order("hit_count", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (exact) return exact;

  // Fallback to neutral personality for same intent
  const { data: neutral } = await supabase
    .from("response_cache")
    .select("id, response_text, response_type, intent_id, personality, source, reviewed, hit_count")
    .eq("case_id", case_id)
    .eq("intent_id", intent_id)
    .eq("personality", "neutral")
    .eq("reviewed", true)
    .order("hit_count", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (neutral) return neutral;

  // Last: accept unreviewed if model_confidence is high enough
  const { data: unreviewed } = await supabase
    .from("response_cache")
    .select("id, response_text, response_type, intent_id, personality, source, reviewed, hit_count, model_confidence")
    .eq("case_id", case_id)
    .eq("intent_id", intent_id)
    .eq("reviewed", false)
    .gte("model_confidence", CONFIDENCE_THRESHOLD)
    .order("model_confidence", { ascending: false })
    .limit(1)
    .maybeSingle();

  return unreviewed ?? null;
}


// ── STAGE 2 HELPER — vector similarity search ───────────────

async function fetchByVector(
  supabase:    ReturnType<typeof createClient>,
  case_id:     string,
  embedding:   number[],
  personality: string,
): Promise<CacheEntry | null> {

  // Fetch top 3 candidates, then pick best by similarity × hit_count weight
  const { data, error: rpcError } = await supabase.rpc("match_response_cache", {
    query_embedding:      embedding,
    match_case_id:        case_id,
    match_personality:    personality,
    similarity_threshold: SIMILARITY_THRESHOLD,
    match_count:          3,
  });

  if (rpcError) {
    console.error("[retrieve] Vector RPC error:", rpcError.message);
    return null;
  }

  if (!data?.length) return null;

  // Score = similarity (weighted 70%) + normalised hit_count (weighted 30%)
  const maxHits = Math.max(...data.map((r: CacheEntry & { similarity: number }) => r.hit_count), 1);
  const best = (data as Array<CacheEntry & { similarity: number }>).reduce((a, b) => {
    const scoreA = (a.similarity * 0.7) + ((a.hit_count / maxHits) * 0.3);
    const scoreB = (b.similarity * 0.7) + ((b.hit_count / maxHits) * 0.3);
    return scoreA >= scoreB ? a : b;
  });

  return best;
}


// ── STAGE 3 HELPER — classic fallback ───────────────────────

function getClassicFallback(
  caseData:    Record<string, unknown>,
  intent_id:   string | null | undefined,
  personality: string = "neutral",
): { text: string; type: string; score: number } {

  if (intent_id) {
    const intentMap = caseData.intent_map as Record<string, { text: string; type: string }>;
    const entry     = intentMap?.[intent_id];
    if (entry) {
      return { text: entry.text, type: entry.type, score: 5 };
    }
  }

  // Personality-aware fallback — improves realism without LLM cost
  const fallbackByPersonality: Record<string, string> = {
    anxious:     "I'm not really sure… I'm a bit worried, can you ask me something else?",
    stoic:       "I don't think that's relevant.",
    reticent:    "...I'm not sure.",
    cooperative: "Hmm, I don't know how to answer that. Maybe try another question?",
    neutral:     "I'm not sure that's relevant to my condition. Could you ask me something else?",
  };

  return {
    text:  fallbackByPersonality[personality] ?? fallbackByPersonality.neutral,
    type:  "history",
    score: 0,
  };
}


// ── LLM CALL ────────────────────────────────────────────────
//  Uses Vertex AI Gemini 2.5 Flash Lite Preview
//  Auth: service account JWT → short-lived access token (cached 55 min)
//  Secret: GOOGLE_SERVICE_ACCOUNT_KEY (base64-encoded JSON)

interface LLMResult {
  cleaned_question:  string;
  answer:            string;
  intent:            string | null;
  keywords:          string[];
  category:          string;
  urgency:           string;
  confidence:        number;
  personality_notes: string;
  flags:             string[];
}

// Module-level token cache — reused across requests in same isolate lifetime
let _tokenCache: { token: string; expiry: number } | null = null;

async function getVertexAccessToken(): Promise<string> {
  const now = Date.now();

  // Return cached token if still valid (55 min window, tokens expire at 60)
  if (_tokenCache && now < _tokenCache.expiry) {
    return _tokenCache.token;
  }

  const b64   = Deno.env.get("GOOGLE_SERVICE_ACCOUNT_KEY")!;
  const creds = JSON.parse(atob(b64));

  const now   = Math.floor(Date.now() / 1000);
  const header = btoa(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const claim  = btoa(JSON.stringify({
    iss:   creds.client_email,
    scope: "https://www.googleapis.com/auth/cloud-platform",
    aud:   "https://oauth2.googleapis.com/token",
    iat:   now,
    exp:   now + 3600,
  }));

  const pemBody   = creds.private_key
    .replace("-----BEGIN PRIVATE KEY-----", "")
    .replace("-----END PRIVATE KEY-----", "")
    .replace(/\n/g, "");
  const binaryDer = Uint8Array.from(atob(pemBody), c => c.charCodeAt(0));

  const privateKey = await crypto.subtle.importKey(
    "pkcs8", binaryDer.buffer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false, ["sign"],
  );

  const signingInput = `${header}.${claim}`;
  const signature    = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    privateKey,
    new TextEncoder().encode(signingInput),
  );

  const jwt = `${signingInput}.${btoa(String.fromCharCode(...new Uint8Array(signature)))}`;

  const tokenResp = await fetch("https://oauth2.googleapis.com/token", {
    method:  "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body:    `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  });

  const tokenData = await tokenResp.json();

  // Cache for 55 minutes
  _tokenCache = {
    token:  tokenData.access_token as string,
    expiry: now + (55 * 60 * 1000),
  };

  return _tokenCache.token;
}

async function callLLM(
  question:    string,
  caseData:    Record<string, unknown>,
  personality: string,
): Promise<LLMResult> {

  const profile = caseData.patient_profile as Record<string, unknown>;

  const systemInstruction = `
You are roleplaying as a patient in a Nigerian teaching hospital OSCE examination.
Patient profile: ${JSON.stringify(profile)}
Case context: ${caseData.case_context ?? ""}
Personality: ${personality}

Respond ONLY with a valid JSON object. No markdown, no backticks, no preamble.
Schema:
{
  "cleaned_question":  "normalised version of the student's question",
  "answer":            "your reply as the patient (10–400 characters)",
  "intent":            "snake_case intent id if you can infer it, else null",
  "keywords":          ["keyword1", "keyword2"],
  "category":          "history" | "exam" | "investigation" | "social",
  "urgency":           "low" | "medium" | "high",
  "confidence":        0.00–1.00,
  "personality_notes": "how personality affected this answer",
  "flags":             [] or ["out_of_scope"] or ["reveals_diagnosis"] or ["meta_language"]
}

Rules:
- Never reveal the diagnosis directly.
- Never break character or mention AI, scenarios, or training.
- Never advise the student on what to do.
- If the question is irrelevant, set flags: ["out_of_scope"] and give a brief dismissive reply.
- Answer must be consistent with the patient profile (age, gender, presenting complaint).
`.trim();

  const projectId = Deno.env.get("GOOGLE_CLOUD_PROJECT") ?? "savvy-scion-476418-q7";
  const location  = "us-central1";
  const model     = "gemini-2.5-flash-lite-preview";
  const url       = `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${model}:generateContent`;

  const accessToken = await getVertexAccessToken();

  const resp = await fetchWithRetry(url, {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      system_instruction: {
        parts: [{ text: systemInstruction }],
      },
      contents: [
        { role: "user", parts: [{ text: question }] },
      ],
      generationConfig: {
        temperature:      0.4,   // low temp for consistent, factual patient replies
        maxOutputTokens:  512,
      },
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Vertex AI API error ${resp.status}: ${errText}`);
  }

  const data  = await resp.json();

  // Vertex AI response shape mirrors Gemini:
  // data.candidates[0].content.parts[0].text
  const raw   = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  const clean = raw.replace(/```json|```/g, "").trim();

  try {
    return JSON.parse(clean) as LLMResult;
  } catch (e) {
    console.error("[retrieve] JSON parse failed. Raw output:", clean);
    // Return safe fallback instead of crashing Stage 4
    return {
      cleaned_question:  question,
      answer:            "I'm not sure I understand. Could you rephrase that?",
      intent:            null,
      keywords:          [],
      category:          "history",
      urgency:           "low",
      confidence:        0,
      personality_notes: "",
      flags:             ["parse_error"],
    };
  }
}


// ── VALIDATION LAYER ────────────────────────────────────────

interface ValidationResult {
  pass:    boolean;
  rule?:   string;
  reason?: string;
}

function validateLLMResponse(
  r:        LLMResult,
  caseData: Record<string, unknown>,
): ValidationResult {

  const profile = caseData.patient_profile as Record<string, unknown>;
  const fail    = (rule: string, reason: string): ValidationResult =>
    ({ pass: false, rule, reason });

  // ── Structural rules
  if (!r || typeof r !== "object")
    return fail("schema", "Response is not an object");

  if (!r.answer || typeof r.answer !== "string")
    return fail("field_presence", "answer field missing or not a string");

  if (!r.category || typeof r.category !== "string")
    return fail("field_presence", "category field missing");

  if (typeof r.confidence !== "number")
    return fail("field_presence", "confidence field missing or not a number");

  if (r.answer.length < 10)
    return fail("answer_length", `Answer too short: ${r.answer.length} chars`);

  if (r.answer.length > 400)
    return fail("answer_length", `Answer too long: ${r.answer.length} chars`);

  const validCategories = ["history", "exam", "investigation", "social"];
  if (!validCategories.includes(r.category))
    return fail("invalid_enum", `Invalid category: ${r.category}`);

  const validUrgencies = ["low", "medium", "high"];
  if (r.urgency && !validUrgencies.includes(r.urgency))
    return fail("invalid_enum", `Invalid urgency: ${r.urgency}`);

  if (r.confidence < 0 || r.confidence > 1)
    return fail("confidence_range", `Confidence out of range: ${r.confidence}`);

  if (!Array.isArray(r.keywords) || r.keywords.length === 0)
    return fail("keywords_empty", "keywords must be a non-empty array");

  // ── Content safety rules
  const metaPhrases = [
    "as an ai", "i'm an ai", "language model", "in this scenario",
    "i cannot", "i'm sorry, i can", "as instructed",
  ];
  const answerLower = r.answer.toLowerCase();

  for (const phrase of metaPhrases) {
    if (answerLower.includes(phrase))
      return fail("meta_language", `Answer contains meta-language: "${phrase}"`);
  }

  // No markdown in answer
  if (/[*#`_]/.test(r.answer))
    return fail("markdown_in_answer", "Answer contains markdown characters");

  // LLM flagged it as out_of_scope — still valid, just noting
  // (we allow out_of_scope, but reject reveals_diagnosis)
  if (r.flags?.includes("reveals_diagnosis"))
    return fail("reveals_diagnosis", "LLM flagged: answer reveals diagnosis");

  // ── Case consistency rules
  const age    = profile?.age as number | undefined;
  const gender = (profile?.gender as string | undefined)?.toLowerCase();

  if (age !== undefined && age < 14) {
    const adultPhrases = ["my husband", "my wife", "i've been working", "my job"];
    for (const phrase of adultPhrases) {
      if (answerLower.includes(phrase))
        return fail("age_inconsistency", `Paediatric case but adult phrase: "${phrase}"`);
    }
  }

  if (gender === "male") {
    const femalePhrases = ["my period", "i'm pregnant", "my pregnancy", "last menstrual"];
    for (const phrase of femalePhrases) {
      if (answerLower.includes(phrase))
        return fail("gender_inconsistency", `Male patient but female phrase: "${phrase}"`);
    }
  }

  if (gender === "female") {
    const malePhrases = ["my prostate", "my testes"];
    for (const phrase of malePhrases) {
      if (answerLower.includes(phrase))
        return fail("gender_inconsistency", `Female patient but male phrase: "${phrase}"`);
    }
  }

  // ── Diagnosis leak check
  const diagnosis = caseData.diagnosis as Record<string, unknown>;
  const diagName  = (diagnosis?.primary as string ?? "").toLowerCase();
  if (diagName && answerLower.includes(diagName))
    return fail("diagnosis_leak", `Answer contains diagnosis name: "${diagName}"`);

  return { pass: true };
}


// ── RATE LIMIT ──────────────────────────────────────────────

async function checkRateLimit(
  supabase:   ReturnType<typeof createClient>,
  session_id: string,
): Promise<{ allowed: boolean }> {

  const now      = Date.now();
  const windowMs = RATE_WINDOW_SECONDS * 1000;

  // Read current session state
  const { data: session } = await supabase
    .from("sessions")
    .select("request_count, last_request_at")
    .eq("session_id", session_id)
    .maybeSingle();

  const lastRequest = session ? new Date(session.last_request_at).getTime() : 0;
  const resetCount  = !session || (now - lastRequest > windowMs);
  const newCount    = resetCount ? 1 : session!.request_count + 1;

  if (newCount > MAX_REQUESTS_PER_SESSION) {
    return { allowed: false };
  }

  // Atomic upsert — eliminates read-modify-write race condition
  await supabase.from("sessions").upsert({
    session_id,
    request_count:   newCount,
    last_request_at: new Date().toISOString(),
  }, { onConflict: "session_id" });

  return { allowed: true };
}


// ── EMBEDDING ───────────────────────────────────────────────
//  Uses Vertex AI text-embedding-004 — 768-dim.
//  Unified under Vertex — same auth as LLM, single billing system.
//  Checks embedding_cache first — avoids re-embedding the
//  same question string. Cache key is the exact text.

async function embedText(
  text:     string,
  supabase: ReturnType<typeof createClient>,
): Promise<number[]> {

  // ── Cache lookup (O(1) PK lookup)
  const { data: cached } = await supabase
    .from("embedding_cache")
    .select("embedding")
    .eq("text", text)
    .maybeSingle();

  if (cached?.embedding) {
    console.log("[retrieve] embedding cache hit");
    return cached.embedding as unknown as number[];
  }

  // ── Call Vertex AI embedding endpoint
  const projectId   = Deno.env.get("GOOGLE_CLOUD_PROJECT") ?? "savvy-scion-476418-q7";
  const location    = "us-central1";
  const model       = "text-embedding-004";
  const url         = `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${model}:predict`;
  const accessToken = await getVertexAccessToken();

  const resp = await fetchWithRetry(url, {
    method:  "POST",
    headers: {
      "Authorization": `Bearer ${accessToken}`,
      "Content-Type":  "application/json",
    },
    body: JSON.stringify({
      instances: [{ content: text, task_type: "RETRIEVAL_QUERY" }],
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Vertex embedding error ${resp.status}: ${errText}`);
  }

  const data      = await resp.json();
  const embedding = data.predictions?.[0]?.embeddings?.values ?? [];

  // ── Write to cache (non-blocking)
  if (embedding.length > 0) {
    supabase.from("embedding_cache")
      .upsert({ text, embedding }, { onConflict: "text" })
      .catch(err => console.warn("[retrieve] embedding cache write failed:", err));
  }

  return embedding;
}


// ── GEMINI WITH RETRY + TIMEOUT ─────────────────────────────
//  Timeout: 8s per attempt — prevents hanging functions.
//  Retry: one retry on transient 5xx failures.
//  Does not retry on 400/401/403 (bad key, bad request).

function fetchWithTimeout(
  url:       string,
  options:   RequestInit,
  timeoutMs  = 8000,
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(id));
}

async function fetchWithRetry(
  url:     string,
  options: RequestInit,
  retries  = 1,
): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const resp = await fetchWithTimeout(url, options);
    if (resp.ok || resp.status < 500 || attempt === retries) return resp;
    console.warn(`[retrieve] Gemini ${resp.status}, retrying (attempt ${attempt + 1})...`);
    await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
  }
  // unreachable but satisfies TS
  return fetchWithTimeout(url, options);
}


// ── HELPERS ─────────────────────────────────────────────────

function normalisePersonality(p: string): string {
  const valid = ["neutral", "stoic", "anxious", "reticent", "cooperative"];
  return valid.includes(p) ? p : "neutral";
}

function getIntentScore(
  caseData:  Record<string, unknown>,
  intent_id: string,
): number {
  const scoring = caseData.scoring_map as Record<string, unknown>;
  if (!scoring) return 5;
  const mustAsk   = (scoring.mustAsk   as string[]) ?? [];
  const shouldAsk = (scoring.shouldAsk as string[]) ?? [];
  if (mustAsk.includes(intent_id))   return (scoring.pointsMust   as number) ?? 20;
  if (shouldAsk.includes(intent_id)) return (scoring.pointsBase   as number) ?? 10;
  return 5;
}

async function checkLLMCap(
  supabase:   ReturnType<typeof createClient>,
  session_id: string,
): Promise<{ allowed: boolean }> {

  const now      = Date.now();
  const windowMs = RATE_WINDOW_SECONDS * 1000;

  const { data: session } = await supabase
    .from("sessions")
    .select("llm_call_count, last_llm_at")
    .eq("session_id", session_id)
    .maybeSingle();

  const lastLLM  = session ? new Date(session.last_llm_at ?? 0).getTime() : 0;
  const resetCount = !session || (now - lastLLM > windowMs);
  const newCount   = resetCount ? 1 : (session!.llm_call_count ?? 0) + 1;

  if (newCount > MAX_LLM_CALLS_PER_SESSION) {
    return { allowed: false };
  }

  // Non-blocking update
  supabase.from("sessions").upsert({
    session_id,
    llm_call_count: newCount,
    last_llm_at:    new Date().toISOString(),
  }, { onConflict: "session_id" }).catch(() => {});

  return { allowed: true };
}


// ── ANALYTICS ───────────────────────────────────────────────
//  Non-blocking — fire and forget. Logs stage, source, latency.
//  Use these counts to monitor cache hit rate and LLM spend.

function trackAnalytics(
  supabase:   ReturnType<typeof createClient>,
  payload: {
    session_id:    string;
    case_id:       string;
    stage:         "exact" | "vector" | "llm" | "fallback" | "classic";
    source:        string;
    latency_ms:    number;
    intent_id:     string | null;
  },
): void {
  supabase.from("analytics_log").insert(payload)
    .catch(err => console.warn("[retrieve] analytics write failed:", err));
}

async function incrementHitCount(
  supabase: ReturnType<typeof createClient>,
  id:       string,
): Promise<void> {
  // Non-blocking — fire and forget, but log failures
  supabase.rpc("increment_hit_count", { entry_id: id })
    .catch(err => console.warn("[retrieve] hit count failed:", err));
}

async function logError(
  supabase: ReturnType<typeof createClient>,
  payload:  {
    case_id:          string;
    raw_question:     string;
    llm_raw_response: string;
    failure_rule:     string;
    failure_detail:   string;
  },
): Promise<void> {
  await supabase.from("error_queue").insert(payload);
}

function respond(payload: RetrieveResponse): Response {
  return new Response(JSON.stringify(payload), {
    status:  200,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

function error(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}


// ============================================================
//  COMPANION SQL — run in Supabase SQL editor
//  Required for Stage 3 vector search (supabase.rpc call)
//
//  NOTE: Gemini text-embedding-004 produces 768-dim vectors.
//  Run these ALTER statements BEFORE creating the function
//  if you already ran the v2 schema (which used 1536-dim):
//
//  ALTER TABLE response_cache ALTER COLUMN embedding TYPE VECTOR(768);
//  ALTER TABLE unmatched_log  ALTER COLUMN embedding TYPE VECTOR(768);
//
//  Also add the duplicate-prevention index for auto-saves:
//
//  CREATE UNIQUE INDEX IF NOT EXISTS unique_auto_cache
//    ON response_cache (case_id, intent_id, personality)
//    WHERE reviewed = FALSE;
// ============================================================
//
// CREATE OR REPLACE FUNCTION match_response_cache(
//   query_embedding      VECTOR(768),
//   match_case_id        TEXT,
//   match_personality    TEXT,
//   similarity_threshold FLOAT,
//   match_count          INT   -- pass 3 from Edge Function
// )
// RETURNS TABLE (
//   id             UUID,
//   response_text  TEXT,
//   response_type  TEXT,
//   intent_id      TEXT,
//   personality    TEXT,
//   source         TEXT,
//   reviewed       BOOLEAN,
//   hit_count      INT,
//   similarity     FLOAT
// )
// LANGUAGE sql STABLE
// AS $$
//   SELECT
//     rc.id,
//     rc.response_text,
//     rc.response_type::TEXT,
//     rc.intent_id,
//     rc.personality::TEXT,
//     rc.source::TEXT,
//     rc.reviewed,
//     rc.hit_count,
//     1 - (rc.embedding <=> query_embedding) AS similarity
//   FROM response_cache rc
//   WHERE
//     rc.case_id    = match_case_id
//     AND (rc.personality::TEXT = match_personality OR rc.personality::TEXT = 'neutral')
//     AND (rc.reviewed = TRUE OR rc.model_confidence >= 0.75)  -- allow high-conf unreviewed
//     AND 1 - (rc.embedding <=> query_embedding) > similarity_threshold
//   ORDER BY rc.embedding <=> query_embedding
//   LIMIT match_count;
// $$;
