// .github/scripts/generate_greeting.js
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";

const githubPatForModels = process.env.GITHUB_PAT_FOR_MODELS;
const endpoint = "https://models.github.ai/inference";
const modelId = "openai/gpt-4.1-mini";

const prAuthor = process.env.PR_AUTHOR;
const prNumber = process.env.PR_NUMBER; 
const prTitle = process.env.PR_TITLE;

async function main() {
  let finalCommentText = `Error: Default fallback for PR #${prNumber} by @${prAuthor}.`; 

  if (!githubPatForModels) {
    console.error("Error: GITHUB_PAT_FOR_MODELS environment variable is not set."); // Use console.error
    finalCommentText = `Thanks for your contribution, @${prAuthor}! (Automated greeting failed: Missing API Key for AI model).`;
    console.log(finalCommentText); 
    return;
  }
  if (!prNumber || !prAuthor || !prTitle) {
    console.error("Error: Missing one or more PR/Repo environment variables for context."); // Use console.error
    finalCommentText = `Thanks for your contribution! (Automated greeting failed: Missing PR context).`;
    console.log(finalCommentText); 
    return;
  }

  console.error(`Processing PR #${prNumber} ('${prTitle}') by @${prAuthor} for comment generation.`); // Use console.error

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
  
  console.error(`Selected user prompt for ${modelId}: ${selectedUserPrompt}`); // Use console.error

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
      console.error("Error from AI model API:", response.body.error); // Use console.error
      finalCommentText = `Thanks for your contribution, @${prAuthor}! (Automated greeting had an AI API issue for PR "${prTitle}").\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
    } else {
      let commentText = response.body.choices[0]?.message?.content || "";
      commentText = commentText.trim();
      
      if (commentText.toLowerCase().startsWith("```text")) { 
        commentText = commentText.substring(7).trim(); 
      } else if (commentText.toLowerCase().startsWith("text")) { 
        commentText = commentText.substring(4).trim(); 
      }
      if (commentText.endsWith("```")) { 
        commentText = commentText.slice(0, -3).trim(); 
      }
      
      if (commentText) {
        finalCommentText = `${commentText}\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
      } else {
        finalCommentText = `Thanks for your great contribution, @${prAuthor}, on PR "${prTitle}"! We'll take a look soon. 👍\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
      }
    }
  } catch (err) {
    console.error("The script encountered an error during AI generation:", err); // Use console.error
    finalCommentText = `Welcome, @${prAuthor}! Thanks for opening PR "${prTitle}". We'll review it shortly. (Automated greeting experienced an issue.)\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
  }

  // This is the ONLY console.log that should print the final comment to STDOUT
  console.log(finalCommentText); 
}

main();
