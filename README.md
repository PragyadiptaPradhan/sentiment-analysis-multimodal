# Multimodal Sentiment Analysis Platform

End-to-end multimodal sentiment/emotion analysis system with:

- a **Next.js web/API layer** for auth, upload, quota management, and inference orchestration
- a **PyTorch + SageMaker ML pipeline** for training and deployment

The system accepts video files, runs speech transcription + multimodal inference (text/audio/video), and returns per-utterance emotion/sentiment predictions.

## Repository Structure

```text
multimodal/
├─ frontend/   # Next.js app (API + UI + auth + DB + S3/SageMaker integration)
└─ model/      # Training and deployment code for the multimodal model
```

### `frontend/` highlights

- Next.js 15 + React 19 + TypeScript
- NextAuth (credentials/auth routes)
- Prisma + SQLite schema (users, sessions, API quotas, uploaded files)
- AWS integrations:
  - pre-signed S3 upload URLs
  - SageMaker Runtime endpoint invocation

Key API routes:

- `POST /api/upload-url` → validates API key, returns pre-signed URL + object key
- `POST /api/sentiment-inference` → validates API key/quota, invokes SageMaker endpoint

### `model/` highlights

- Multimodal model training pipeline (text + audio + video)
- SageMaker training launcher (`train_sagemaker.py`)
- SageMaker endpoint deployer (`deploy_endpoint.py`)
- Inference handler (`deployment/inference.py`) used by SageMaker PyTorchModel

Inference request payload expected by endpoint code:

```json
{
  "video_path": "s3://<bucket>/<key>"
}
```

Inference response shape (simplified):

```json
{
  "utterances": [
    {
      "start_time": 0.0,
      "end_time": 2.1,
      "text": "...",
      "emotions": [{ "label": "happiness", "confidence": 0.82 }],
      "sentiments": [{ "label": "positive", "confidence": 0.76 }]
    }
  ]
}
```

## Prerequisites

## General

- Node.js (recommended: 20+)
- npm
- Python 3.11+
- AWS account with access to:
  - S3 bucket for uploaded videos
  - SageMaker endpoint/runtime
  - IAM roles and credentials used by training/deployment

## Frontend

- SQLite (default via Prisma datasource)

## Model

- CUDA-capable GPU recommended for training/inference workloads
- FFmpeg available in runtime environments (training and deployment scripts handle setup attempts)

## Quick Start

## 1) Frontend setup

```bash
cd frontend
npm install
```

Create `frontend/.env` (or `.env.local`) with required variables:

```env
AUTH_SECRET=your-auth-secret
DATABASE_URL=file:./db.sqlite
NODE_ENV=development

AWS_REGION=your-region
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_INFERENCE_BUCKET=your-s3-bucket-or-arn
AWS_ENDPOINT_NAME=your-sagemaker-endpoint-name
```

Initialize Prisma and run dev server:

```bash
npm run db:push
npm run dev
```

Useful scripts:

- `npm run dev` - start local development server
- `npm run build` - production build
- `npm run start` - run production server
- `npm run lint` / `npm run typecheck` / `npm run check`

## 2) Model setup (local Python environment)

```bash
cd ../model
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You can also install scoped dependencies from:

- `model/training/requirements.txt` (training)
- `model/deployment/requirements.txt` (inference container)

## Running Training and Deployment

## Local training entrypoint

```bash
cd model/training
python train.py --epochs 20 --batch-size 16 --learning-rate 0.001
```

Default SageMaker/local channel paths used by `train.py`:

- training csv: `train_sent_emo.csv`
- validation csv: `dev_sent_emo.csv`
- test csv: `test_sent_emo.csv`

Expected split directories:

- `train_splits`
- `dev_splits_complete`
- `output_repeated_splits_test`

## Start SageMaker training job

```bash
cd model
python train_sagemaker.py
```

## Deploy SageMaker endpoint

```bash
cd model
python deploy_endpoint.py
```

Note: the current scripts contain hardcoded IAM roles, S3 paths, and endpoint/model names. Update those values for your AWS account before running.

## Data and Docs

- Dataset notes: `model/dataset/README.txt`
- Training docs: `model/training/docs/`

## End-to-End Flow

1. Client gets API key (stored in `ApiQuota.secretKey` in DB).
2. Client calls `POST /api/upload-url` with `Authorization: Bearer <apiKey>`.
3. Frontend returns a pre-signed S3 URL and object key.
4. Client uploads video directly to S3.
5. Client calls `POST /api/sentiment-inference` with API key and uploaded object key.
6. Frontend validates ownership/quota and invokes SageMaker endpoint.
7. Endpoint returns utterance-level emotion/sentiment predictions.

## Troubleshooting

- `npm run dev` fails immediately:
  - verify all required env vars in `frontend/.env`
  - run `npm install` and `npm run db:push`
- Prisma connection issues:
  - ensure `DATABASE_URL` points to a writable SQLite file path
- SageMaker invocation errors:
  - check `AWS_REGION`, `AWS_INFERENCE_BUCKET`, `AWS_ENDPOINT_NAME`
  - verify IAM permissions for `s3:*` (needed scope) and `sagemaker:InvokeEndpoint`
- Inference audio/video failures:
  - confirm FFmpeg availability in training/inference runtime

## Security Notes

- Do not commit real AWS credentials or production secrets.
- Rotate API keys and IAM credentials regularly.
- Restrict IAM policies to least privilege for S3 and SageMaker access.
