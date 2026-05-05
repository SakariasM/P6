@echo off
del /f /q gt_mask_progress.pkl 2>nul
del /f /q gt_mask_prompts.pkl 2>nul
rmdir /s /q gt_mask_masks 2>nul
echo Cleared checkpoints and masks.
