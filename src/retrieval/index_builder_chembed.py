"""Build the separate Chroma collection chembed_index."""
from __future__ import annotations
import argparse, hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import chromadb, yaml
from .embedder import load_embedder, encode_texts, model_metadata

ROOT = Path(__file__).resolve().parents[2]
def path_from_root(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p
def unique(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists(): return p
    for n in range(1,10000):
        q=p.with_name(f"{p.stem}_{n:02d}{p.suffix}")
        if not q.exists(): return q
    raise RuntimeError("Cannot allocate unique output filename")
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/retrieval_chembed.yaml")
    ap.add_argument("--reset",action="store_true")
    ap.add_argument("--limit",type=int)
    a=ap.parse_args()
    cfg=yaml.safe_load(path_from_root(a.config).read_text(encoding="utf-8"))
    src=path_from_root(cfg["document"]["input_path"])
    records=[json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit is not None:
        if a.limit < 1: ap.error("--limit must be >= 1")
        records=records[:a.limit]
    template=cfg["document"]["text_template"]; idkey=cfg["document"].get("id_field","unit_id")
    docs=[]; seen=set(); skipped=0
    for row in records:
        rid=row.get(idkey)
        if not isinstance(rid,str) or not rid.strip(): raise ValueError(f"Missing {idkey}")
        if rid in seen: raise ValueError(f"Duplicate ID: {rid}")
        seen.add(rid)
        if not str(row.get("text","")).strip():
            if cfg["document"].get("skip_empty_text",True): skipped+=1; continue
            raise ValueError(f"Empty text: {rid}")
        docs.append((rid,template.format(**row).strip(),row))
    if not docs: raise RuntimeError("No indexable records")
    vs=cfg["vector_store"]
    client=chromadb.HttpClient(host=vs.get("host","localhost"),port=int(vs.get("port",8000)))
    cname=vs.get("collection_name","chembed_index")
    if a.reset:
        try: client.delete_collection(cname)
        except Exception: pass
    col=client.get_or_create_collection(cname,metadata={"hnsw:space":vs.get("distance_metric","cosine")})
    metric=(col.metadata or {}).get("hnsw:space")
    if metric and metric != vs.get("distance_metric","cosine"):
        raise RuntimeError(f"Collection metric {metric} != configured metric; use --reset")
    model=load_embedder(cfg); batch=int(vs.get("upsert_batch_size",32)); done=0
    for i in range(0,len(docs),batch):
        group=docs[i:i+batch]
        vectors=encode_texts(model,[x[1] for x in group],cfg,
            identifiers=[x[0] for x in group],task="document")
        metas=[]
        for _,_,row in group:
            meta={}
            for k,v in row.items():
                if v is None: continue
                if isinstance(v,(str,int,float,bool)): meta[k]=v
                else: meta[k]=json.dumps(v,ensure_ascii=False)
            metas.append(meta or {"source":"ncert"})
        col.upsert(ids=[x[0] for x in group],embeddings=vectors.tolist(),
                   documents=[x[1] for x in group],metadatas=metas)
        done+=len(group); print(f"Indexed {done}/{len(docs)}")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest={"built_at_utc":datetime.now(timezone.utc).isoformat(),
        "input_path":str(src),"input_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
        "records_read":len(records),"records_indexed_this_run":done,
        "records_skipped_empty":skipped,"collection_name":col.name,
        "collection_count_after_build":col.count(),
        "distance_metric":vs.get("distance_metric","cosine"),
        "embedding":model_metadata(model),"text_template":template,
        "python_version":platform.python_version(),"chromadb_version":chromadb.__version__,
        "reset_requested":a.reset}
    outdir=path_from_root(cfg.get("outputs",{}).get("manifest_dir","reports/retrieval/chembed"))
    out=unique(outdir/f"index_manifest_{stamp}.json")
    out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Collection {col.name}: {col.count()} records\nManifest: {out}")
if __name__=="__main__": main()
