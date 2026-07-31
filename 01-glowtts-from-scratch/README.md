# Audio AI — GlowTTS

A simple Text-to-Speech model training project using Coqui TTS and LJSpeech.

## Goal

Understand and build a complete TTS training pipeline.

## what i built

- started from a raw ljspeech dataset and built the full pipeline from audio files and transcripts to a trainable glowtts model.

- worked through coqui tts's config-based architecture to understand how dataset loading, audio processing, tokenization, model setup, and training are connected.

- i found coqui tts's shared configuration architecture is harder to debug because components inherit settings through the same config object rather than exposing their own isolated interfaces.

- debugged a failed training run where most audio samples were being skipped, traced the issue back to an incomplete dataset extraction, and rebuilt the pipeline with a clean dataset.

- trained glowtts on google colab t4 gpu and learned how to read training signals like loss, duration loss, gradient values, and checkpoints through tensorboard.

- built the project as a reproducible notebook workflow that can be resumed from checkpoints and extended towards inference.

## Stack

- Model: GlowTTS
- Framework: Coqui TTS
- Dataset: LJSpeech
- Compute: Google Colab T4 GPU

## Pipeline

```text
text + audio dataset
        ↓
dataset formatter
        ↓
audio processor
        ↓
tokenizer
        ↓
glowtts training
        ↓
checkpoint
        ↓
inference
```
```
