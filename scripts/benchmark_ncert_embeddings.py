"""Raw dense retrieval benchmark; run separately for MiniLM and ChEmbed."""
from __future__ import annotations
import argparse,json,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import chromadb,yaml
from src.retrieval.embedder import load_embedder,encode_texts,model_metadata
ROOT=Path(__file__).resolve().parents[1]
def rooted(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p
def unique(p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists(): return p
    for n in range(1,10000):
        q=p.with_name(f"{p.stem}_{n:02d}{p.suffix}")
        if not q.exists(): return q
    raise RuntimeError("Cannot allocate unique output")
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/retrieval.yaml")
    ap.add_argument("--benchmark",default="chemistry_retrieval_benchmark_v1.json")
    ap.add_argument("--label",required=True)
    ap.add_argument("--top-k",type=int,default=10)
    ap.add_argument("--query-mode",choices=["question_and_response","question"],default="question_and_response")
    ap.add_argument("--output-dir",default="artifacts/retrieval_benchmark")
    a=ap.parse_args()
    if a.top_k<1: ap.error("--top-k must be >= 1")
    cfg=yaml.safe_load(rooted(a.config).read_text(encoding="utf-8"))
    bp=rooted(a.benchmark)
    if not bp.exists():
        candidate=ROOT/"src/retrieval"/Path(a.benchmark).name
        if candidate.exists(): bp=candidate
        else: raise FileNotFoundError(bp)
    bench=json.loads(bp.read_text(encoding="utf-8"))
    if not isinstance(bench.get("queries"),list): raise ValueError("Benchmark needs queries[]")
    vs=cfg["vector_store"]
    client=chromadb.HttpClient(host=vs.get("host","localhost"),port=int(vs.get("port",8000)))
    col=client.get_collection(vs["collection_name"])
    if not col.count(): raise RuntimeError(f"Collection {col.name} is empty")
    model=load_embedder(cfg); results=[]; lat=[]
    for item in bench["queries"]:
        qid=str(item.get("query_id") or f"Q{len(results)+1:02d}")
        question=str(item.get("question") or "").strip()
        response=str(item.get("student_response") or "").strip()
        query=question if a.query_mode=="question" else f"Student question: {question}\nStudent response: {response}".strip()
        t=time.perf_counter()
        vector=encode_texts(model,[query],cfg,identifiers=[qid],task="query")[0]
        found=col.query(query_embeddings=[vector.tolist()],n_results=min(a.top_k,col.count()),
                        include=["documents","metadatas","distances"])
        ms=(time.perf_counter()-t)*1000; lat.append(ms)
        hits=[]
        for rank,(rid,doc,meta,dist) in enumerate(zip(found["ids"][0],found["documents"][0],
            found["metadatas"][0],found["distances"][0]),1):
            hits.append({"rank":rank,"id":rid,"distance":float(dist),
                         "document":doc or "","metadata":meta or {}})
        results.append({"query_id":qid,"query_text":query,
            "target_concepts":item.get("target_concepts",[]),"latency_ms":round(ms,3),"hits":hits})
        print(f"{qid}: {len(hits)} chunks")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output={"benchmark_id":bench.get("benchmark_id"),"run_timestamp_utc":stamp,
        "model_label":a.label,"config_path":str(rooted(a.config)),"benchmark_path":str(bp),
        "query_mode":a.query_mode,"top_k":a.top_k,"collection_name":col.name,
        "collection_count":col.count(),"embedding":model_metadata(model),"results":results,
        "latency_ms":{"mean":round(statistics.mean(lat),3),"p50":round(statistics.median(lat),3),
                      "p95":round(sorted(lat)[min(len(lat)-1,int(.95*(len(lat)-1)))],3)},
        "evaluation_status":"raw_unjudged",
        "notes":["Raw dense retrieval; no reranking.","Distances are not relevance probabilities.",
                 "Unjudged IDs must not be scored as irrelevant."]}
    out=unique(rooted(a.output_dir)/f"{a.label}_chunks_{stamp}.json")
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Retrieved chunks saved: {out}")
if __name__=="__main__": main()
