import requests,sys,json
def q(t,rows=3):
    import time
    for _ in range(4):
        try:
            r=requests.get("https://api.crossref.org/works",params={"query.bibliographic":t,"rows":rows,"select":"DOI,title,author,container-title,issued,volume,issue,page,type"},timeout=30).json()["message"]["items"]; break
        except Exception: time.sleep(4)
    else: print("  FAIL"); return
    for i in r:
        a=", ".join(x.get("family","") for x in i.get("author",[])[:4])
        print(" ",i["DOI"],"|",(i.get("title") or ["?"])[0][:90],"|",a,"|",(i.get("container-title") or [""])[0][:40],"|",i.get("issued",{}).get("date-parts",[[None]])[0][0],i.get("volume"),i.get("page"))
for t in open(sys.argv[1]).read().strip().split("\n"):
    print("##",t); q(t)
