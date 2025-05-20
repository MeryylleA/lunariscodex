// .github/scripts/generate_greeting.js
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
// import { execSync } from 'child_process'; // No longer needed

const githubPatForModels = process.env.GITHUB_PAT_FOR_MODELS;
const endpoint = "https://models.github.ai/inference";
const modelId = "openai/gpt-4.1-mini";

const prAuthor = process.env.PR_AUTHOR;
const prNumber = process.env.PR_NUMBER; // Still useful for logging/context
const prTitle = process.env.PR_TITLE;
// const repoFullName = process.env.REPO_FULL_NAME; // No longer needed by this script
// const ghTokenForComment = process.env.GH_TOKEN_FOR_COMMENT; // No longer needed by this script

async function main() {
  if (!githubPatForModels) {
    console.error("Error: GITHUB_PAT_FOR_MODELS environment variable is not set.");
    // Instead of process.exit(1), we'll let the workflow handle fallback
    console.log("Failed to generate unique greeting: Missing PAT"); 
    return; // Exit function, workflow will use fallback
  }
  // ... other env var checks can also just return or print an error message to be captured ...

  console.log(`Processing PR #${prNumber} ('${prTitle}') by @${prAuthor} for comment generation.`);

  const client = ModelClient(
    endpoint,
    new AzureKeyCredential(githubPatForModels),
  );

  const greetingPrompts = [
      `Write a friendly and welcoming GitHub pull request comment for a new PR titled "${prTitle}" by user @${prAuthor}. Thank them for their contribution to Lunaris Codex. Keep it concise (around 2-3 sentences) and encouraging, and mention that their PR will be reviewed soon. Add a relevant emoji at the end.`,
      `Craft a brief, positive comment for GitHub PR #${prNumber} ("${prTitle}") from @${prAuthor}. Express appreciation for their effort on Lunaris Codex and state that the maintainers will look at it shortly. Use a welcoming tone and an emoji. Max 3 sentences.`,
      `Generate a warm thank you message for @${prAuthor} for submitting the pull request "${prTitle}" to the Lunaris Codex project. Let them know their contribution is valued and will be reviewed. Include a friendly emoji. Keep it short.`,
  ];
  const selectedSystemPrompt = "You are a friendly assistant for the Lunaris Codex open-source project, tasked with welcoming new pull requests.";
  const selectedUserPrompt = greetingPrompts[Math.floor(Math.random() * greetingPrompts.length)];
  
  console.log(`Selected user prompt for ${modelId}: ${selectedUserPrompt}`);
  let finalCommentText = ""; // Initialize

  try {
    const response = await client.path("/chat/completions").post({
      body: {
        messages: [
          { role: "system", content: selectedSystemPrompt },
          { role: "user", content: selectedUserPrompt }
        ],
        temperature: 0.75,
        top_p: 0.9,
        max_tokens: 150,
        model: modelId
      }
    });

    if (isUnexpected(response)) {
      console.error("Error from AI model API:", response.body.error);
      finalCommentText = `Failed to generate a unique greeting due to API error (PR: ${prTitle} by @${prAuthor}).`;
    } else {
      let commentText = response.body.choices[0]?.message?.content || "";
      commentText = commentText.trim();
      // Basic cleanup
      if (commentText.lower().startswith("```text")) { commentText = commentText.substring(7).trim(); }
      else if (commentText.lower().startswith("text")) { commentText = commentText.substring(4).trim(); }
      if (commentText.endswith("```")) { commentText = commentText.slice(0, -3).trim(); }
      
      if (commentText) {
        finalCommentText = `${commentText}\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
      } else {
        finalCommentText = `Failed to generate a unique greeting, but thanks for your PR, @${prAuthor}! (PR: ${prTitle})`;
      }
    }
  } catch (err) {
    console.error("The script encountered an error during AI generation:", err);
    finalCommentText = `Error during comment generation for PR "${prTitle}" by @${prAuthor}.`;
  }

  // Output the final comment text to STDOUT for the GitHub Action to capture
  // This is the crucial part for passing it to the next step.
  console.log(finalCommentText); 
}

main(); // Removed .catch here, workflow will see script success/failure by exit code.
        // If main throws an unhandled error, the script will exit non-zero.
