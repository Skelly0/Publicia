# Google Check Grounding API Setup Guide

## Overview

Publicia now supports Google's Vertex AI Check Grounding API for validating AI responses against source documents. This provides enterprise-grade grounding validation with detailed citation tracking and contradiction detection.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud Project** with appropriate APIs enabled
3. **API credentials** (API key or service account)

## Setup Steps

### 1. Create/Configure Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Note your **Project ID** (you'll need this)

### 2. Enable Required APIs

Enable these APIs in your Google Cloud project:

```bash
# Using gcloud CLI
gcloud services enable discoveryengine.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Or enable via Cloud Console:
# - Discovery Engine API
# - Vertex AI API
```

### 3. Set Up Authentication

**Important**: The Google Discovery Engine API requires OAuth2 authentication and does not support API keys directly.

#### Service Account Setup (Required)
1. Go to **IAM & Admin** → **Service Accounts**
2. Create service account with these roles:
   - `Discovery Engine Admin`
   - `Vertex AI User`
3. Generate JSON key file
4. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your JSON key file

**Note**: API keys alone are not sufficient for the Discovery Engine API. You must use service account credentials.

### 4. Configure Publicia

Add these environment variables to your `.env` file:

```env
# Required for Google Check Grounding API
GOOGLE_PROJECT_ID=your-google-cloud-project-id

# Required: Service account credentials file path
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Usage limits (optional - defaults shown)
GROUNDING_MAX_DAILY_CHECKS=1000
GROUNDING_COST_PER_CHECK=0.001
GROUNDING_MAX_DAILY_BUDGET=1.0

# Note: GOOGLE_API_KEY is not used for Discovery Engine API
# but may be required for other Google services in the project
GOOGLE_API_KEY=your-google-api-key
```

### 5. Verify Setup

Test the grounding API with:

```bash
python test_grounding.py
```

If configured correctly, you should see:
- ✅ Google API credentials detected
- ✅ API calls successful
- Detailed grounding scores and citations

## API Features

### Core Functionality
- **Support Score**: 0.0-1.0 indicating how well response is grounded
- **Claim Analysis**: Sentence-level grounding validation
- **Citation Mapping**: Links claims to supporting facts
- **Contradiction Detection**: Identifies conflicting information

### Advanced Features (Experimental)
- **Anti-Citations**: Detect contradictory facts
- **Helpfulness Score**: Evaluate response quality
- **Claim-Level Scores**: Individual sentence validation

## Usage in Discord

Once configured, grounding checks automatically work with:

- **`/query` commands**: Automatic grounding display
- **Mention responses**: When user has grounding enabled
- **Manual checks**: `/check_grounding` command

### Example Output

```
🟢 **Grounding Check**: Well-grounded (3/3 claims supported, 95% confidence)
📊 Support: 0.95 | Helpfulness: 0.87 | Claims: 3 supported, 0 contradicted
```

### Usage Monitoring

Check your API usage with:
- **`/grounding_usage`**: View daily usage statistics
- **Automatic limits**: Prevents exceeding daily quotas
- **Fallback behavior**: Uses local implementation when limits reached

## Usage Limits & Cost Control

### Built-in Safeguards

Publicia includes automatic usage limits to prevent unexpected costs:

- **Daily Check Limit**: Default 1000 checks per day
- **Daily Budget Limit**: Default $1.00 per day
- **Automatic Fallback**: Uses local implementation when limits reached
- **Usage Tracking**: Persistent tracking across bot restarts

### Configuration Options

```env
# Maximum API calls per day
GROUNDING_MAX_DAILY_CHECKS=1000

# Cost per API call (for budget tracking)
GROUNDING_COST_PER_CHECK=0.001

# Maximum daily budget in USD
GROUNDING_MAX_DAILY_BUDGET=1.0
```

### Monitoring Commands

- **`/grounding_usage`**: Check current usage and remaining quota
- **Real-time tracking**: Usage logged with each API call
- **Daily reset**: Limits reset at midnight UTC

## Troubleshooting

### Common Issues

1. **"No Google Project ID provided"**
   - Add `GOOGLE_PROJECT_ID` to `.env` file
   - Ensure project exists and APIs are enabled

2. **"API request failed with status 403"**
   - Check API key permissions
   - Verify APIs are enabled in project
   - Check billing is enabled

3. **"API request failed with status 404"**
   - Verify project ID is correct
   - Ensure Discovery Engine API is enabled

4. **Fallback to local implementation**
   - This is normal when API is not configured
   - Local implementation provides basic grounding using text similarity

### Getting Help

1. Check [Google Cloud Discovery Engine documentation](https://cloud.google.com/discovery-engine/docs)
2. Verify your [Google Cloud billing](https://console.cloud.google.com/billing)
3. Review [API quotas and limits](https://cloud.google.com/discovery-engine/quotas)

## Cost Considerations

- Check Grounding API pricing: ~$0.001 per request
- Typical usage: 10-50 requests per day for active Discord server
- Monthly cost estimate: $0.30-$1.50 for moderate usage

## Security Best Practices

1. **Restrict API Keys**: Limit to specific APIs and IP ranges
2. **Use Service Accounts**: For production deployments
3. **Monitor Usage**: Set up billing alerts
4. **Rotate Keys**: Regularly update API credentials

## Fallback Behavior

If Google API is unavailable, Publicia automatically falls back to:
- Local text similarity computation
- Basic contradiction detection
- Simplified scoring system

This ensures grounding functionality always works, even without API access.
