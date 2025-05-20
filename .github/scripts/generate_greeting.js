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
  let fallbackMessage = `Thanks for your contribution, @${prAuthor}! We'll review your PR titled "${prTitle}" soon. ✨`;
  let finalCommentText = ""; 

  if (!githubPatForModels) {
    console.error("CRITICAL: GITHUB_PAT_FOR_MODELS environment variable is not set. Cannot call AI model.");
    finalCommentText = `${fallbackMessage}\n\n---\n*Lunaris Codex Assistant (fallback: missing API Key).*`;
    console.log(finalCommentText);
    return;
  }
  if (!prNumber || !prAuthor || !prTitle) {
    console.error("CRITICAL: Missing one or more PR/Repo environment variables for context.");
    finalCommentText = `${fallbackMessage}\n\n---\n*Lunaris Codex Assistant (fallback: missing PR context).*`;
    console.log(finalCommentText);
    return;
  }

  console.error(`Attempting to generate greeting for PR #${prNumber} ('${prTitle}') by @${prAuthor} using ${modelId}.`);

  const client = ModelClient(
    endpoint,
    new AzureKeyCredential(githubPatForModels),
  );

  let systemMessage = "";
  let userPrompts = [];

  // Determine prompts based on author
  if (prAuthor === "dependabot[bot]") {
    systemMessage = "You are the Lunaris Codex project assistant, with a slightly witty and familiar tone when addressing the Dependabot. You appreciate its diligence but find its constant updates a bit amusing.";
    userPrompts = [
      `Dependabot opened PR "${prTitle}". Generate a short, playful comment acknowledging its hard work, like "Ah, Dependabot, my old friend! Always on the ball with these updates. We'll check this one out. Thanks! 🤖"`,
      `Craft a comment for Dependabot's PR "${prTitle}". Tone: "Look who it is! Our ever-vigilant Dependabot keeping things fresh. Good bot! Review incoming. 👍"`,
      `Write a message for Dependabot about PR "${prTitle}". Something like: "And there it is, right on schedule! Dependabot strikes again. Thanks for the dependency bump, we're on it! 😉"`
    ];
  } else if (prAuthor === "MeryylleA" || prAuthor === "lunaris-codex-bot") { // Your account and your bot account
    systemMessage = "You are the Lunaris Codex project assistant, acting as a supportive and encouraging AI pair programmer for the lead developer, @MeryylleA.";
    userPrompts = [
      `@${prAuthor} submitted PR "${prTitle}". Write an encouraging comment: "Nice one, @${prAuthor}! This looks like an interesting update. Diving into the details soon. Keep up the great work! ✨"`,
      `Comment on PR "${prTitle}" by @${prAuthor}: "Another great contribution, @${prAuthor}! Excited to see what's new this time. Review will commence shortly. 🚀"`,
      `Acknowledge PR "${prTitle}" from @${prAuthor}: "Thanks for pushing this through, @${prAuthor}! Your work on Lunaris Codex is much appreciated. Taking a look now... 🧐"`
    ];
  } else { // Default for other contributors
    systemMessage = "You are a friendly and welcoming assistant for the Lunaris Codex open-source project, tasked with greeting new pull requests from the community.";
    userPrompts = [
      `Write a friendly and welcoming GitHub pull request comment for a new PR titled "${prTitle}" by user @${prAuthor}. Thank them for their contribution to Lunaris Codex. Keep it concise (around 2-3 sentences), encouraging, and mention that their PR will be reviewed soon. Add a relevant emoji at the end.`,
      `Craft a brief, positive comment for GitHub PR #${prNumber} ("${prTitle}") from @${prAuthor}. Express appreciation for their effort on Lunaris Codex and state that the maintainers will look at it shortly. Use a welcoming tone and an emoji. Max 3 sentences.`,
      `Generate a warm thank you message for @${prAuthor} for submitting the pull request "${prTitle}" to the Lunaris Codex project. Let them know their contribution is valued and will be reviewed. Include a friendly emoji. Keep it short and cheerful.`
    ];
  }
  
  const selectedUserPrompt = userPrompts[Math.floor(Math.random() * userPrompts.length)];
  console.error(`Selected system prompt: ${systemMessage}`);
  console.error(`Selected user prompt for LLM: ${selectedUserPrompt}`);

  try {
    const response = await client.path("/chat/completions").post({
      body: {
        messages: [
          { role: "system", content: systemMessage },
          { role: "user", content: selectedUserPrompt }
        ],
        temperature: 0.8, // Slightly higher for more creative/varied responses
        top_p: 0.9,
        max_tokens: 180, // Increased slightly for potentially more expressive greetings
        model: modelId 
      }
    });

    if (isUnexpected(response) || !response.body.choices || response.body.choices.length === 0 || !response.body.choices[0].message?.content) {
      console.error("Error or empty response from AI model API:", response.body?.error || "Empty choices/content");
      finalCommentText = fallbackMessage; // Use the predefined fallback
    } else {
      let aiGeneratedText = response.body.choices[0].message.content.trim();
      
      if (aiGeneratedText.toLowerCase().startsWith("```text")) { aiGeneratedText = aiGeneratedText.substring(7).trim(); }
      else if (aiGeneratedText.toLowerCase().startsWith("text")) { aiGeneratedText = aiGeneratedText.substring(4).trim(); }
      if (aiGeneratedText.endsWith("```")) { aiGeneratedText = aiGeneratedText.slice(0, -3).trim(); }
      
      if (aiGeneratedText) {
        finalCommentText = aiGeneratedText;
        console.error(`Successfully generated message from AI: ${finalCommentText}`);
      } else {
        console.error("AI generated an empty message after cleanup. Using fallback.");
        finalCommentText = fallbackMessage;
      }
    }
  } catch (err) {
    console.error("Exception during AI model call:", err);
    finalCommentText = fallbackMessage; // Use fallback on any exception during AI call
  }

  const commentBodyWithSignature = `${finalCommentText}\n\n---\n*This is an automated greeting from the Lunaris Codex Assistant. 🤖*`;
  console.log(commentBodyWithSignature); // This is the final output to STDOUT
}

main();
