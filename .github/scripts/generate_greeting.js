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
  // Default fallback message - this will be overwritten by AI if successful
  let generatedMessage = `Thanks for your contribution, @${prAuthor}! We'll review your PR titled "${prTitle}" soon. ✨`;

  if (!githubPatForModels) {
    console.error("CRITICAL: GITHUB_PAT_FOR_MODELS environment variable is not set. Cannot call AI model.");
    // Outputting the fallback directly so the workflow step can capture it.
    console.log(`${generatedMessage}\n\n---\n*Lunaris Codex Assistant (fallback: missing API Key).*`);
    return;
  }
  if (!prNumber || !prAuthor || !prTitle) {
    console.error("CRITICAL: Missing one or more PR/Repo environment variables for context.");
    console.log(`${generatedMessage}\n\n---\n*Lunaris Codex Assistant (fallback: missing PR context).*`);
    return;
  }

  console.error(`Attempting to generate greeting for PR #${prNumber} ('${prTitle}') by @${prAuthor} using ${modelId}.`);

  const client = ModelClient(
    endpoint,
    new AzureKeyCredential(githubPatForModels),
  );

  const greetingPrompts = [
      `Write a friendly and welcoming GitHub pull request comment for a new PR titled "${prTitle}" by user @${prAuthor}. Thank them for their contribution to Lunaris Codex. Keep it concise (around 2-3 sentences), encouraging, and mention their PR will be reviewed soon. Add a relevant emoji at the end.`,
      `Craft a brief, positive comment for GitHub PR #${prNumber} ("${prTitle}") from @${prAuthor}. Express appreciation for their effort on Lunaris Codex and state that the maintainers will look at it shortly. Use a welcoming tone and an emoji. Maximum 3 sentences.`,
      `Generate a warm thank you message for @${prAuthor} for submitting the pull request "${prTitle}" to the Lunaris Codex project. Let them know their contribution is valued and will be reviewed. Include a friendly emoji. Keep it short and cheerful.`,
  ];
  const systemMessage = "You are a friendly assistant for the Lunaris Codex open-source project, designed to write welcoming comments for new pull requests. Be encouraging and positive.";
  const userMessage = greetingPrompts[Math.floor(Math.random() * greetingPrompts.length)];
  
  console.error(`Selected user prompt for LLM: ${userMessage}`);

  try {
    const response = await client.path("/chat/completions").post({
      body: {
        messages: [
          { role: "system", content: systemMessage },
          { role: "user", content: userMessage }
        ],
        temperature: 0.75,
        top_p: 0.9,
        max_tokens: 150, // Max output tokens for the comment
        model: modelId 
      }
    });

    if (isUnexpected(response) || !response.body.choices || response.body.choices.length === 0 || !response.body.choices[0].message?.content) {
      console.error("Error or empty response from AI model API:", response.body?.error || "Empty choices/content");
      // Keep the pre-defined fallback in generatedMessage
    } else {
      let aiGeneratedText = response.body.choices[0].message.content.trim();
      
      // Basic cleanup for common LLM artifacts
      if (aiGeneratedText.toLowerCase().startsWith("```text")) { 
        aiGeneratedText = aiGeneratedText.substring(7).trim(); 
      } else if (aiGeneratedText.toLowerCase().startsWith("text")) { 
        aiGeneratedText = aiGeneratedText.substring(4).trim(); 
      }
      if (aiGeneratedText.endsWith("```")) { 
        aiGeneratedText = aiGeneratedText.slice(0, -3).trim(); 
      }
      
      if (aiGeneratedText) { // If AI provided a non-empty cleaned response
        generatedMessage = aiGeneratedText;
        console.error(`Successfully generated message from AI: ${generatedMessage}`);
      } else {
        console.error("AI generated an empty message after cleanup. Using fallback.");
        // Fallback already set in generatedMessage
      }
    }
  } catch (err) {
    console.error("Exception during AI model call:", err);
    // Fallback already set in generatedMessage
  }

  // Add the standard signature to whatever message we have (AI-generated or fallback)
  const finalCommentBody = `${generatedMessage}\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
  
  // This is the only output that should go to STDOUT for the GitHub Action
  console.log(finalCommentBody);
}

main();
