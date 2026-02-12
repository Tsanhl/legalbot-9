<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1FqeNKsRJ7SV-iHiCS98y1nXHChPirUcW

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
# legal-doc
 
# legal-bot

## ChromaDB on Hugging Face (recommended)

GitHub should not store your large local DB (`chroma_db/` is ~6GB).  
Use a Hugging Face dataset repo for Chroma persistence.

### 1) Upload local ChromaDB to HF dataset

```bash
# login once (if needed)
hf auth login

# upload only chroma_db contents (not the full project)
python3 scripts/sync_chromadb_hf.py push --repo-id Agnes999/legalbot9
```

### 2) Auto-pull DB at startup (Render/local)

Set environment variables:

```bash
CHROMA_HF_DATASET_REPO=Agnes999/legalbot9
CHROMA_HF_AUTO_PULL=1
HF_TOKEN=hf_xxx   # required if dataset is private
```

Optional:

```bash
CHROMA_PERSIST_DIR=/opt/render/project/src/chroma_db
CHROMA_HF_FORCE_PULL=0
CHROMA_HF_REVISION=main
```

`rag_service.py` now auto-downloads from HF on startup if local DB is missing.

## Can I delete `Law resouces  copy 2`?

Yes, for retrieval-only runtime, it is not required once the index is fully built and `chroma_db` is safely backed up (HF dataset).  
Keep a backup if you may need to re-index later.

## Quick checks

```bash
python3 scripts/sync_chromadb_hf.py status
python3 -c "from rag_service import get_rag_service; r=get_rag_service(); print('chunks', r.collection.count()); print('ctx', len(r.get_relevant_context('contract consideration', max_chunks=3)))"
```
