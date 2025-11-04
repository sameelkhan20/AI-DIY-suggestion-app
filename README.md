---
title: AI Image Analyzer
emoji: 🖼️
colorFrom: purple
colorTo: blue
sdk: docker
sdk_version: 3.0.0
app_file: Dockerfile
pinned: false
license: mit
---

# AI Image Analyzer

A powerful web application that uses OpenAI's Vision API to analyze images and provide creative reuse suggestions, DIY ideas, and monetization opportunities.

## Features

- 🎯 **Single Image Analysis**: Upload individual images for detailed AI-powered analysis
- 📸 **Camera Capture**: Take photos directly from your device for instant analysis
- 📦 **Batch Processing**: Process multiple images simultaneously
- 💡 **Smart Recommendations**: Get DIY ideas, monetization suggestions, and sustainability benefits
- 🎨 **Creative Reuse**: Discover upcycling and transformation opportunities

## How to Use

1. Upload an image or take a photo using the camera feature
2. Wait for AI analysis (powered by OpenAI GPT-4o Vision)
3. View detailed analysis including:
   - Object identification
   - Material analysis
   - Condition assessment
   - Creative reuse suggestions
   - Monetization opportunities
   - Sustainability benefits

## Setup

### Environment Variables

You need to set the following secret in Hugging Face Spaces:

- `OPENAI_API_KEY`: Your OpenAI API key (required for AI analysis)

To set secrets:
1. Go to your Space settings
2. Navigate to "Variables and secrets"
3. Add `OPENAI_API_KEY` as a secret

## Technology Stack

- **Backend**: Flask (Python)
- **AI**: OpenAI GPT-4o Vision API
- **Image Processing**: Pillow (PIL)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## License

MIT License
