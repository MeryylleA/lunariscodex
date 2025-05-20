// .github/scripts/generate_greeting.js
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import { execSync } from 'child_process'; // To run gh cli

const githubPat = process.env.GITHUB_PAT_FOR_MODELS;
const endpoint = "https://models.github.ai/inference";
const modelId = "openai/gpt-4.1-mini"; // Correct model ID

const prAuthor = process.env.PR_AUTHOR;
const prNumber = process.env.PR_NUMBER;
const prTitle = process.env.PR_TITLE;
const repoFullName = process.env.REPO_FULL_NAME;
const ghTokenForComment = process.env.GH_TOKEN_FOR_COMMENT; // For gh cli

async function main() {
    if (!githubPat) {
        console.error("Error: GITHUB_PAT_FOR_MODELS environment variable is not set.");
        process.exit(1);
    }
    if (!prNumber || !prAuthor || !prTitle || !repoFullName || !ghTokenForComment) {
        console.error("Error: Missing one or more PR/Repo environment variables for context or commenting.");
        process.exit(1);
    }

    console.log(`Processing PR #${prNumber} ('${prTitle}') by @${prAuthor}`);

    const client = ModelClient(
        endpoint,
        new AzureKeyCredential(githubPat),
    );

    // --- Prompt Engineering ---
    const greetingPrompts = [
        `Write a friendly and welcoming GitHub pull request comment for a new PR titled "${prTitle}" by user @${prAuthor}. Thank them for their contribution to Lunaris Codex. Keep it concise (around 2-3 sentences) and encouraging, and mention that their PR will be reviewed soon. Add a relevant emoji at the end.`,
        `Craft a brief, positive comment for GitHub PR #${prNumber} ("${prTitle}") from @${prAuthor}. Express appreciation for their effort on Lunaris Codex and state that the maintainers will look at it shortly. Use a welcoming tone and an emoji. Max 3 sentences.`,
        `Generate a warm thank you message for @${prAuthor} for submitting the pull request "${prTitle}" to the Lunaris Codex project. Let them know their contribution is valued and will be reviewed. Include a friendly emoji. Keep it short.`,
    ];
    const selectedSystemPrompt = "You are a friendly assistant for the Lunaris Codex open-source project, tasked with welcoming new pull requests.";
    const selectedUserPrompt = greetingPrompts[Math.floor(Math.random() * greetingPrompts.length)];

    console.log(`Selected user prompt for ${modelId}: ${selectedUserPrompt}`);

    try {
        const response = await client.path("/chat/completions").post({
            body: {
                messages: [
                    { role: "system", content: selectedSystemPrompt },
                    { role: "user", content: selectedUserPrompt }
                ],
                temperature: 0.75, // For a bit of variety
                top_p: 0.9,
                max_tokens: 150, // Max output tokens for the comment
                model: modelId // Pass the model ID here
            }
        });

        if (isUnexpected(response)) {
            console.error("Error from AI model API:", response.body.error);
            throw response.body.error;
        }

        let commentText = response.body.choices[0]?.message?.content || "Failed to generate a unique greeting, but thanks for your PR!";

        // Basic cleanup
        commentText = commentText.trim();

        // Add a standard signature
        commentText += `\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;

        console.log(`Generated comment: ${commentText}`);

        // Post the comment using gh CLI
        // Ensure GITHUB_TOKEN in workflow has 'pull-requests: write' on pull_request_target or use a PAT for ghTokenForComment
        // For security, escape the comment text for the shell command
        const escapedCommentText = commentText.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\$/g, '\\$');
        const ghCommand = `gh issue comment ${prNumber} --repo ${repoFullName} --body "${escapedCommentText}"`;

        console.log("Executing gh command to post comment...");
        execSync(ghCommand, { stdio: 'inherit', env: { ...process.env, GITHUB_TOKEN: ghTokenForComment } });
        console.log("Comment posted successfully.");

    } catch (err) {
        console.error("The script encountered an error:", err);
        // Optionally, post a fallback static comment if AI fails
        const fallbackComment = `Thanks for your contribution, @${prAuthor}! We'll review your PR titled "${prTitle}" soon. ✨\n\n---\n*Lunaris Codex Assistant (fallback message).*`;
        const escapedFallbackComment = fallbackComment.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\$/g, '\\$');
        const fallbackGhCommand = `gh issue comment ${prNumber} --repo ${repoFullName} --body "${escapedFallbackComment}"`;
        try {
            console.log("Attempting to post fallback comment...");
            execSync(fallbackGhCommand, { stdio: 'inherit', env: { ...process.env, GITHUB_TOKEN: ghTokenForComment } });
            console.log("Fallback comment posted.");
        } catch (fallbackErr) {
            console.error("Failed to post fallback comment:", fallbackErr);
        }
    }
}

main().catch((err) => {
    console.error("Unhandled error in main:", err);
    process.exit(1); // Exit with error on unhandled exception
});
