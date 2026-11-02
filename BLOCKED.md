# BLOCKED log

Entries are appended by scripts/alarm.ps1 whenever human action is needed.

- [2026-11-03 00:16:02] Alarm self-test (CrisisLens) - you can click OK
- [2026-11-03 00:20:18] NEED: (1) Sign up at adaptionlabs.ai/app/auth, create API key at adaptionlabs.ai/app/settings, paste into D:\DEvharbour\.env as ADAPTION_API_KEY= ; tell me your credit balance. (2) HF write token -> HF_TOKEN= in same file. (3) Optional: Together AI key -> TOGETHER_API_KEY= (faster eval). Reply in chat when done.
- [2026-11-03 01:25:42] TOGETHER key rejected (401). Get key at api.together.ai/settings/api-keys (starts tgp_v1_ or long hex), replace TOGETHER_API_KEY in .env. Together may need ~5 USD credit for serving the fine-tuned model - your call. HF inference returned 402 (no credits).
- [2026-11-03 02:33:40] Evals killed by Claude Code memory-pressure guard (two llama-servers ~7.7GB). Need user OK to restart them one at a time (say: restart evals). Alarm sound skipped because user said they are asleep.
