/**
 * Janglish recordings Worker.
 *
 * Routes:
 *   GET  /health                      → 200
 *   GET  /assignment?t=<token>        → dev stub JSON (R2 wiring comes next)
 *   PUT  /upload?t=<token>&id=<id>    → accepts audio/wav body, returns {bytes}
 *
 * Tokens are base64url(payload).base64url(hmacSHA256(secret, payload_b64))
 * where payload is compact JSON {"name":"alice","exp":<unix>}. The Go side
 * (internal/token) signs with the same secret and format.
 */

export interface Env {
  JANGLISH_HMAC_SECRET: string;
  /** Comma-separated list of allowed origins for CORS. Defaults to "*" for dev. */
  ALLOWED_ORIGIN?: string;
  RECORDINGS: R2Bucket;
}

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(req, env) });
    }

    if (req.method === "GET" && url.pathname === "/health") {
      return withCORS(new Response("ok", { status: 200 }), req, env);
    }

    if (req.method === "GET" && url.pathname === "/assignment") {
      return withCORS(await handleAssignment(url, env), req, env);
    }

    if (req.method === "PUT" && url.pathname === "/upload") {
      return withCORS(await handleUpload(req, url, env), req, env);
    }

    return withCORS(new Response("not found", { status: 404 }), req, env);
  },
} satisfies ExportedHandler<Env>;

async function handleAssignment(url: URL, env: Env): Promise<Response> {
  const token = url.searchParams.get("t") ?? "";
  const claims = await verifyToken(token, env.JANGLISH_HMAC_SECRET);
  if (!claims) return new Response("unauthorized", { status: 401 });

  const key = `assignments/${claims.name}.json`;
  const obj = await env.RECORDINGS.get(key);
  if (obj === null) return new Response("assignment not found", { status: 404 });

  return new Response(obj.body, {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

async function handleUpload(req: Request, url: URL, env: Env): Promise<Response> {
  const token = url.searchParams.get("t") ?? "";
  const id = url.searchParams.get("id") ?? "";
  const claims = await verifyToken(token, env.JANGLISH_HMAC_SECRET);
  if (!claims) return new Response("unauthorized", { status: 401 });
  if (!id) return new Response("missing id", { status: 400 });
  if (!/^[A-Za-z0-9_-]+$/.test(id)) return new Response("invalid id", { status: 400 });

  const ctype = req.headers.get("content-type") ?? "";
  if (!ctype.includes("audio/wav") && !ctype.includes("audio/wave")) {
    return new Response("expected audio/wav", { status: 415 });
  }

  const body = await req.arrayBuffer();
  if (body.byteLength > MAX_UPLOAD_BYTES) {
    return new Response("payload too large", { status: 413 });
  }

  const key = `manual/${claims.name}/${id}.wav`;
  await env.RECORDINGS.put(key, body, {
    httpMetadata: { contentType: "audio/wav" },
  });

  return Response.json({
    name: claims.name,
    id,
    bytes: body.byteLength,
    key,
  });
}

// --- token verification (mirrors internal/token in Go) ---

interface Claims {
  name: string;
  exp: number;
}

async function verifyToken(token: string, secret: string): Promise<Claims | null> {
  if (!token || !secret) return null;
  const dot = token.indexOf(".");
  if (dot < 0 || dot === token.length - 1) return null;
  const payloadB64 = token.slice(0, dot);
  const sigB64 = token.slice(dot + 1);

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const macBytes = new Uint8Array(
    await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payloadB64)),
  );
  const want = b64urlEncode(macBytes);
  if (!timingSafeEqual(want, sigB64)) return null;

  let claims: Claims;
  try {
    const payload = new TextDecoder().decode(b64urlDecode(payloadB64));
    claims = JSON.parse(payload);
  } catch {
    return null;
  }
  if (typeof claims.name !== "string" || typeof claims.exp !== "number") return null;
  if (!/^[A-Za-z0-9][A-Za-z0-9-]*$/.test(claims.name)) return null;
  if (claims.exp * 1000 < Date.now()) return null;
  return claims;
}

function b64urlEncode(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? 0 : 4 - (s.length % 4);
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat(pad);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// --- CORS ---

function corsHeaders(req: Request, env: Env): HeadersInit {
  const origin = req.headers.get("origin") ?? "";
  const allow = (env.ALLOWED_ORIGIN ?? "*").split(",").map((s) => s.trim());
  const ok = allow.includes("*") || allow.includes(origin);
  return {
    "access-control-allow-origin": ok ? (origin || "*") : "",
    "access-control-allow-methods": "GET, PUT, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    vary: "origin",
  };
}

function withCORS(res: Response, req: Request, env: Env): Response {
  const headers = new Headers(res.headers);
  for (const [k, v] of Object.entries(corsHeaders(req, env))) {
    if (typeof v === "string") headers.set(k, v);
  }
  return new Response(res.body, { status: res.status, headers });
}
