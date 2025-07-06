# Channel Tracking and Archiving Feature

This document outlines the functionality of the channel tracking and archiving feature, which allows administrators to keep a persistent, updated log of channel conversations within Publicia's knowledge base.

## Overview

The channel tracking feature introduces a mechanism to automatically archive messages from a specified Discord channel and keep the archive updated with new messages. This is useful for logging important conversations, roleplay sessions, or any channel where a persistent record is desired for future reference and querying.

## Commands

Two new slash commands have been introduced for managing channel tracking:

### `/track_channel`

-   **Description**: Starts tracking a specified channel. When initiated, this command performs an initial archive of the channel's recent history and creates a new document in the knowledge base. It then schedules the channel for periodic updates.
-   **Usage**: `/track_channel [channel] [update_interval_hours]`
-   **Parameters**:
    -   `channel` (Required): The Discord channel to be tracked.
    -   `update_interval_hours` (Optional): The interval in hours at which the bot will check for new messages. Defaults to 6 hours.
-   **Permissions**: Administrator only.

### `/untrack_channel`

-   **Description**: Stops tracking a specified channel. This command removes the channel from the periodic update schedule. The existing archive document will remain in the knowledge base but will no longer receive new updates.
-   **Usage**: `/untrack_channel [channel]`
-   **Parameters**:
    -   `channel` (Required): The Discord channel to stop tracking.
-   **Permissions**: Administrator only.

## Background Task

-   A background task runs automatically at a configurable interval (defaulting to every 6 hours).
-   For each tracked channel, the task fetches all new messages sent since the last update.
-   The new messages are formatted and appended to the end of the channel's corresponding archive document.
-   After appending the new content, the document's search embeddings are regenerated to ensure that the latest messages are immediately available for querying via commands like `/query`.

## How It Works

1.  **Initiation**: An admin uses `/track_channel` to select a channel for tracking.
2.  **Initial Archive**: The bot creates a new document (e.g., `channel_archive_[channel_name]_[timestamp].txt`) and saves the current channel history to it. A unique UUID is assigned to this document.
3.  **Tracking Storage**: The channel's ID, the associated document UUID, and the last message ID are stored in a `tracked_channels.json` file.
4.  **Periodic Updates**: The background task reads `tracked_channels.json` and, for each entry, fetches new messages from the channel using the last saved message ID as a starting point.
5.  **Appending Content**: New messages are appended to the document file identified by the stored UUID.
6.  **Embedding Update**: The entire document is re-processed to update its search embeddings. If `CHANNEL_CONTEXTUALIZATION_ENABLED` is `true`, channel archives are contextualized during this process so each chunk is summarized before embedding. Disable this environment variable to skip contextualization for channel logs.
7.  **Untracking**: Using `/untrack_channel` removes the corresponding entry from `tracked_channels.json`, effectively stopping the update cycle for that channel.
