// server.js
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const deployedVercelFrontendUrl = 'https://ai-sql-client.vercel.app/';
const allowedOrigins = process.env.NODE_ENV === 'production' 
    ? [deployedVercelFrontendUrl] 
    // For local development, allow React dev port and also your deployed frontend for easy testing
    : ['http://localhost:3000', 'http://localhost:3001', deployedVercelFrontendUrl]; 
const { Groq } = require('groq-sdk');
const { createClient } = require('@supabase/supabase-js');

// --- Transformers.js ---
const { pipeline, env: xenovaEnv } = require('@xenova/transformers');
xenovaEnv.allowLocalModels = true;
xenovaEnv.useBrowserCache = false;

// --- Configuration ---
const PORT = process.env.PORT || 3002;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!GROQ_API_KEY || !SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error("ERROR: Missing necessary environment variables. Check .env file.");
    process.exit(1);
}

// --- Initialize Clients ---
const groq = new Groq({ apiKey: GROQ_API_KEY });
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
 
// --- Load Augmented Schema ---
let augmentedSchemaString = '';
try {
    const schemaPath = path.join(__dirname, 'chinook_augmented_schema.json');
    if (!fs.existsSync(schemaPath)) throw new Error(`Schema file not found at ${schemaPath}`);
    const schemaFileContents = fs.readFileSync(schemaPath, 'utf8');
    augmentedSchemaString = JSON.stringify(JSON.parse(schemaFileContents), null, 2);
} catch (error) {
    console.error("ERROR: Could not load/parse augmented_schema.json.", error);
    process.exit(1);
}

// --- Embedding Model Singleton ---
let embedder;
const EMBEDDING_MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
async function getEmbedder() {
    if (embedder === undefined) {
        console.log(`[${new Date().toISOString()}] Attempting to load embedding model: ${EMBEDDING_MODEL_NAME}...`);
        try {
            embedder = await pipeline('feature-extraction', EMBEDDING_MODEL_NAME);
            console.log(`[${new Date().toISOString()}] Embedding model loaded successfully.`);
        } catch (error) {
            console.error(`[${new Date().toISOString()}] FATAL: Failed to load embedding model:`, error);
            embedder = null;
        }
    }
    return embedder;
}
getEmbedder().catch(err => console.error("Initial embedder load failed, RAG will be disabled.", err));

// --- Express App Setup ---
const app = express();
app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (like mobile apps or curl requests) ONLY IN DEVELOPMENT
    // For production, you might want to be stricter and disallow no-origin requests.
    if (!origin && process.env.NODE_ENV !== 'production') {
        console.log("CORS: Allowing request with no origin (development mode or tool)");
        return callback(null, true);
    }
    
    // Check if the provided origin is in our list of allowed origins
    if (allowedOrigins.includes(origin)) {
      console.log(`CORS: Origin ${origin} allowed.`);
      callback(null, true); // Allow
    } else {
      console.warn(`CORS: Origin ${origin} NOT allowed. Allowed are: ${allowedOrigins.join(', ')}`);
      callback(new Error(`Not allowed by CORS. Origin: ${origin}`)); // Block
    }
  },
  methods: ['GET', 'POST', 'OPTIONS'], // Ensure OPTIONS is allowed for preflight requests
  allowedHeaders: ['Content-Type', 'Authorization'], // Add any other headers your frontend might send
  credentials: true // If you ever use cookies or authorization headers
}));

// Handle preflight OPTIONS requests for all routes
app.options('*', cors({
    origin: function (origin, callback) { // Same origin check as above
        if (!origin && process.env.NODE_ENV !== 'production') {
            return callback(null, true);
        }
        if (allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error(`Not allowed by CORS. Origin: ${origin}`));
        }
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));
// Handle preflight OPTIONS requests for all routes
app.options('*', cors({
    origin: function (origin, callback) { // Same origin check as above
        if (!origin && process.env.NODE_ENV !== 'production') {
            return callback(null, true);
        }
        if (allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error(`Not allowed by CORS. Origin: ${origin}`));
        }
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));
app.use(express.json());

// --- API Endpoint to Ask Questions ---
app.get('/ask', async (req, res) => {
    const userQuestion = req.query.question;
    const sessionId = req.query.sessionId || `fallback_session_${Date.now()}`;

    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] === New Request ===`);
    console.log(`[${timestamp}] Session ID: ${sessionId}, Question: "${userQuestion}"`);

    if (!userQuestion || typeof userQuestion !== 'string' || userQuestion.trim() === '') {
        return res.status(400).json({
            question: userQuestion || null,
            naturalLanguageSummary: "Error: A question is required.",
            error: "Missing or invalid 'question' query parameter.",
            sessionId: sessionId
        });
    }

    let relevantHistoryText = "";
    const currentEmbedder = await getEmbedder();

    if (currentEmbedder) {
        try {
            // ... (RAG logic - generating embedding for current question, searching history - KEEP THIS AS IS) ...
            console.log(`[${timestamp}] Generating embedding for current question...`);
            const queryEmbeddingOutput = await currentEmbedder(userQuestion, { pooling: 'mean', normalize: true });
            const queryEmbedding = Array.from(queryEmbeddingOutput.data);

            console.log(`[${timestamp}] Searching chat history for session: ${sessionId}`);
            const { data: relevantHistory, error: searchError } = await supabase.rpc('match_chat_history', {
                p_session_id: sessionId,
                p_query_embedding: queryEmbedding,
                p_match_threshold: 0.72, 
                p_match_count: 3
            });

            if (searchError) {
                console.error(`[${timestamp}] Error searching chat history:`, searchError.message);
            } else if (relevantHistory && relevantHistory.length > 0) {
                console.log(`[${timestamp}] Retrieved ${relevantHistory.length} relevant history items.`);
                relevantHistoryText = "For context, here are relevant parts of our previous conversation:\n" +
                    relevantHistory
                        .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
                        .map(h => `User asked: "${h.user_question}"\nYou previously answered: "${h.nl_summary}"`)
                        .join("\n---\n");
                console.log(`[${timestamp}] History context prepended to prompt:\n${relevantHistoryText.substring(0, 300)}...`);
            } else {
                console.log(`[${timestamp}] No sufficiently relevant chat history found for this session.`);
            }
        } catch (ragError) {
            console.error(`[${timestamp}] Error in RAG process:`, ragError);
        }
    } else {
        console.warn(`[${timestamp}] Embedder not available. Skipping RAG (history retrieval).`);
    }

    // ***** MODIFIED SQL PROMPT *****
    const modelForSqlGeneration = "llama3-70b-8192";
    const sqlPrompt = `${relevantHistoryText ? relevantHistoryText + "\n\nConsidering the above context, " : ""}
You are an expert PostgreSQL data analyst. Your SOLE TASK is to convert the CURRENT user question into a single, valid, executable PostgreSQL query based on the provided database schema description.
DO NOT provide any explanations, apologies, or conversational text before or after the SQL query.
Your entire response MUST be ONLY the SQL query itself.
If you cannot generate a query from the given schema for the question, or if the question is unanswerable with SQL, respond with only the text: 'NO_QUERY_POSSIBLE'

The schema contains tables and columns with non-obvious 'actual_name's; you MUST use these 'actual_name's in the generated SQL.
Rely on the 'semantic_name' and 'description' fields to map user questions to the correct database entities.
Pay close attention to data types, relationships, and 'general_hints' for dirty data or transformations.
If the user asks for a count or a single aggregate value, alias the result column (e.g., SELECT COUNT(*) AS total_count).
If previous conversation context is provided, pay close attention to pronouns (like 'it', 'them', 'these') and references to entities or filters mentioned in earlier turns. The "Current User Question" should be interpreted in light of this context to ensure continuity.
Ensure all subqueries are syntactically correct and do not include unnecessary aliases if used directly in expression contexts.

Database Schema Description:
${augmentedSchemaString}

Current User Question: ${userQuestion}

PostgreSQL Query:`;
    // ***** END OF MODIFIED SQL PROMPT *****

    let generatedSql = null;
    let queryResults = null;
    let naturalLanguageSummary = "Could not process the request."; // Default for unexpected issues

    try {
        console.log(`[${timestamp}] Asking ${modelForSqlGeneration} to generate SQL...`);
        const sqlCompletion = await groq.chat.completions.create({
            messages: [{ role: "user", content: sqlPrompt }],
            model: modelForSqlGeneration, temperature: 0.0,
        });
        generatedSql = sqlCompletion.choices[0]?.message?.content?.trim();

        if (!generatedSql) {
            naturalLanguageSummary = "The AI failed to generate an SQL query for your question.";
            throw new Error("LLM did not return SQL.");
        }

        // ***** HANDLE 'NO_QUERY_POSSIBLE' *****
        if (generatedSql.toUpperCase() === 'NO_QUERY_POSSIBLE') {
            naturalLanguageSummary = "I'm sorry, I couldn't determine an SQL query to answer that question based on the available data schema.";
            console.log(`[${timestamp}] LLM responded with NO_QUERY_POSSIBLE.`);
            // Send response immediately as no SQL execution or further NLG is needed
            return res.json({
                question: userQuestion,
                naturalLanguageSummary: naturalLanguageSummary,
                generatedSql: null, // Explicitly null
                databaseResults: null, // No database interaction
                sessionId: sessionId
            });
        }
        // ***** END OF 'NO_QUERY_POSSIBLE' HANDLING *****

        console.log(`[${timestamp}] Generated SQL (raw from LLM):\n${generatedSql}`);
        const sqlForExecution = generatedSql.replace(/;\s*$/, ""); // Strip trailing semicolon
        console.log(`[${timestamp}] SQL for execution (semicolon stripped):\n${sqlForExecution}`);

        console.log(`[${timestamp}] Executing SQL on Supabase...`);
        const { data: dbExecData, error: dbExecError } = await supabase.rpc('execute_dynamic_sql', {
            query_string: sqlForExecution
        });

        if (dbExecError) { // Error calling the RPC function itself
            console.error(`[${timestamp}] Supabase RPC execution error (calling function):`, dbExecError);
            naturalLanguageSummary = `A system error occurred with the database function: ${dbExecError.message}.`;
            throw new Error(naturalLanguageSummary); // Let the main catch block handle response
        }
        
        queryResults = dbExecData;
        console.log(`[${timestamp}] Raw results from execute_dynamic_sql:`, JSON.stringify(queryResults).substring(0, 500) + (JSON.stringify(queryResults).length > 500 ? '...' : ''));

        // Check if the content of dbExecData indicates an error from *within* the executed SQL
        if (typeof queryResults === 'object' && queryResults !== null && queryResults.error) {
            naturalLanguageSummary = `Database reported an error: ${queryResults.error.message || 'Unknown database error'}. Query: ${queryResults.error.query_attempted || sqlForExecution}`;
            console.error(`[${timestamp}] Error from execute_dynamic_sql (SQL failed inside function):`, queryResults.error);
            // We will still try to store this as a "failed" interaction if embedder is available,
            // but the naturalLanguageSummary will reflect the error.
            // The client will see this error in naturalLanguageSummary.
            // No need to throw here if we want to proceed to store history.
        } else {
            console.log(`[${timestamp}] SQL processed by database function successfully.`);
        }
        

        // Generate Natural Language Summary (even if queryResults contains a DB error message)
        const modelForNLG = "llama3-8b-8192";
        let dataForNlgPrompt = queryResults;
        let shouldCallNlgLLM = false;

        // If queryResults indicates a direct error from the DB function, use that as the summary
        if (typeof queryResults === 'object' && queryResults !== null && queryResults.error) {
            naturalLanguageSummary = `The database query failed. Error: ${queryResults.error.message || JSON.stringify(queryResults.error)}`;
        } else if (Array.isArray(queryResults)) {
            if (queryResults.length === 0) naturalLanguageSummary = "The query ran successfully but returned no data.";
            else shouldCallNlgLLM = true;
        } else if (typeof queryResults === 'object' && queryResults !== null && queryResults.status === 'success' && queryResults.message) {
            naturalLanguageSummary = queryResults.message;
        } else if (queryResults !== null && queryResults !== undefined) { // Some other non-error, non-array, non-status object
            shouldCallNlgLLM = true; // Attempt to summarize whatever it is
        } else { // queryResults is null or undefined (shouldn't happen if Supabase func is robust)
            naturalLanguageSummary = "The query was processed, but no clear results were returned from the database.";
        }


        if (shouldCallNlgLLM) {
            const nlgPrompt = `User's question: "${userQuestion}"
SQL executed: "${sqlForExecution}"
Data (JSON, truncated): ${JSON.stringify(dataForNlgPrompt, null, 2).substring(0, 3000)}${JSON.stringify(dataForNlgPrompt, null, 2).length > 3000 ? "..." : ""}
Provide a concise, natural language answer based on this. If the data indicates an error or is unusual, explain that.
Answer:`;
            console.log(`[${timestamp}] Asking ${modelForNLG} for NL summary...`);
            const nlgCompletion = await groq.chat.completions.create({
                messages: [{ role: "user", content: nlgPrompt }],
                model: modelForNLG, temperature: 0.3,
            });
            naturalLanguageSummary = nlgCompletion.choices[0]?.message?.content?.trim() || "LLM failed to provide a summary based on the data.";
        }
        console.log(`[${timestamp}] Final Natural Language Summary: ${naturalLanguageSummary}`);

        // Store Chat History (even if naturalLanguageSummary is an error message from DB)
        if (currentEmbedder && userQuestion && generatedSql) { // Store if we have user Q and generated SQL
            const textToEmbedForHistory = `User: ${userQuestion}\nSQL: ${generatedSql}\nAgent Answer: ${naturalLanguageSummary}`;
            try {
                console.log(`[${timestamp}] Generating embedding for history turn...`);
                const historyEmbeddingOutput = await currentEmbedder(textToEmbedForHistory, { pooling: 'mean', normalize: true });
                const historyEmbedding = Array.from(historyEmbeddingOutput.data);

                console.log(`[${timestamp}] Storing chat history for session ${sessionId}...`);
                const { error: historyInsertError } = await supabase
                    .from('chat_conversations')
                    .insert({
                        session_id: sessionId, user_question: userQuestion,
                        generated_sql: generatedSql, nl_summary: naturalLanguageSummary, // Store whatever summary we have
                        embedding: historyEmbedding
                    });
                if (historyInsertError) console.error(`[${timestamp}] Error inserting chat history:`, historyInsertError.message);
                else console.log(`[${timestamp}] Chat history turn stored.`);
            } catch (historyEmbedError) {
                console.error(`[${timestamp}] Error embedding/storing chat history:`, historyEmbedError);
            }
        } else if (!currentEmbedder) {
            console.warn(`[${timestamp}] Embedder not available. Skipping history storage.`);
        }

        res.json({
            question: userQuestion, naturalLanguageSummary, generatedSql,
            databaseResults: queryResults === undefined ? null : queryResults,
            sessionId
        });

    } catch (error) { // Main catch block for errors like LLM failure, RPC call failure, etc.
        console.error(`[${timestamp}] ERROR in /ask endpoint processing:`, error.message, error.stack);
        // If naturalLanguageSummary hasn't been set to a specific error message yet, use the caught error's message.
        if (naturalLanguageSummary === "Could not process the request." || !naturalLanguageSummary.toLowerCase().includes("error")) {
            naturalLanguageSummary = error.message || "An unexpected server error occurred.";
        }
        res.status(500).json({ // Default to 500 for unexpected errors caught here
            question: userQuestion, naturalLanguageSummary, generatedSql,
            databaseResults: queryResults, // May be null or contain error from DB function
            error: error.name || "ServerError", details: error.message,
            sessionId
        });
    }
});

// --- Start Server ---
app.listen(PORT, () => {
    console.log(`\nAI SQL Engine server running on http://localhost:${PORT}`);
    console.log("Ensure .env, augmented_schema.json are correct & Supabase functions/extensions are set up.");
    console.log("Waiting for requests...\n");
});
