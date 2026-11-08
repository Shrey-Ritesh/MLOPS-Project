"""Union retrieved IDs and preserve existing MiniLM judgements."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
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
def read(p): return json.loads(rooted(p).read_text(encoding="utf-8"))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minilm",required=True); ap.add_argument("--chembed",required=True)
    ap.add_argument("--existing-annotations",default="src/retrieval/annotated_chunks_minilM.json")
    ap.add_argument("--output-dir",default="artifacts/retrieval_benchmark/pooled_annotations")
    a=ap.parse_args(); m=read(a.minilm); c=read(a.chembed)
    ann_path=rooted(a.existing_annotations)
    ann=json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    old={x["query_id"]:x for x in ann.get("results",[])}
    bm={x["query_id"]:x for x in m.get("results",[])}
    bc={x["query_id"]:x for x in c.get("results",[])}
    if set(bm)!=set(bc): raise ValueError("MiniLM and ChEmbed query IDs differ")
    results=[]
    for qid in sorted(bm):
        prior=old.get(qid,{}).get("corpora",{}).get("ncert",{}).get("hits",[])
        grades={str(x["id"]):int(x["relevance_score"]) for x in prior if x.get("relevance_score") is not None}
        union={}
        for label,row in (("minilm",bm[qid]),("chembed",bc[qid])):
            for hit in row["hits"]:
                rid=str(hit["id"])
                x=union.setdefault(rid,{"id":rid,"relevance_grade":grades.get(rid),
                    "judgement_status":"judged" if rid in grades else "needs_review",
                    "retrieved_by":[],"minilm_rank":None,"chembed_rank":None,
                    "minilm_distance":None,"chembed_distance":None,
                    "document":hit.get("document",""),"metadata":hit.get("metadata",{})})
                x["retrieved_by"].append(label); x[f"{label}_rank"]=hit.get("rank")
                x[f"{label}_distance"]=hit.get("distance")
        results.append({"query_id":qid,"query_text":bm[qid].get("query_text"),
            "target_concepts":bm[qid].get("target_concepts",[]),
            "annotation_scale":{"0":"Not relevant","1":"Supporting context/prerequisite",
                                "2":"Direct evidence for evaluating/explaining response"},
            "hits":sorted(union.values(),key=lambda x:min(v for v in
                (x["minilm_rank"],x["chembed_rank"]) if v is not None))})
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result={"benchmark_id":m.get("benchmark_id"),"created_at_utc":stamp,
        "minilm_run":str(rooted(a.minilm)),"chembed_run":str(rooted(a.chembed)),
        "existing_annotations":str(ann_path) if ann_path.exists() else None,
        "results":results,
        "instructions":"Manually grade every needs_review hit 0/1/2. Null is unknown, never 0."}
    out=unique(rooted(a.output_dir)/f"ncert_pooled_annotations_{stamp}.json")
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(out)
if __name__=="__main__": main()
