#let ink=rgb("18222d")
#let muted=rgb("5d6b7b")
#let gold=rgb("ae7728")
#let pale=rgb("f3f5f7")
#let setup(title,subtitle,body)={
  set document(title:title,author:"LLMIUM",date:datetime(year:2026,month:11,day:28))
  set page(paper:"a4",margin:(top:20mm,bottom:20mm,x:20mm),header:context {
    if counter(page).get().first()>1 {text(size:8pt,fill:muted)[LLMIUM / #subtitle]}
  },footer:context {line(length:100%,stroke:0.4pt+rgb("d9dfe5"));v(5pt);grid(columns:(1fr,auto),text(size:7.5pt,fill:muted)[LOCAL FILES / LOCAL INFERENCE],text(size:7.5pt,fill:muted)[#counter(page).display() / #counter(page).final().first()])})
  set text(font:"Manrope",size:10pt,fill:ink,lang:"en")
  set par(leading:0.65em,justify:false)
  set heading(numbering:none)
  show heading.where(level:1):it=>{v(5pt);text(size:24pt,weight:"bold",it.body);v(8pt)}
  show heading.where(level:2):it=>{v(7pt);text(size:14pt,weight:"bold",it.body);v(3pt)}
  show heading.where(level:3):it=>{v(5pt);text(size:10.5pt,weight:"bold",it.body);v(2pt)}
  show raw: set text(font:"IBM Plex Mono",size:8pt)
  show raw.where(block:true):block.with(fill:pale,inset:11pt,radius:4pt,width:100%)
  show link: set text(fill:gold)
  set list(indent:12pt,body-indent:6pt,spacing:7pt)
  set enum(indent:14pt,body-indent:6pt,spacing:8pt)
  body
}
#let label(body)=text(size:8pt,weight:"bold",fill:gold,tracking:1pt,upper(body))
#let note(title,body)=block(width:100%,fill:rgb("fbf6ed"),stroke:(left:2pt+gold),inset:13pt,radius:3pt)[#text(weight:"bold",title) #v(5pt) #text(size:9pt,fill:muted,body)]
#let card(title,body)=block(width:100%,fill:pale,inset:15pt,radius:5pt)[#text(weight:"bold",title) #v(7pt) #text(size:9pt,fill:muted,body)]
#let cover(kind,title,description)=[
  #label("LLMIUM / MODEL INTEGRITY LAB")
  #v(25mm)
  #text(size:10pt,fill:gold,kind)
  #v(6mm)
  #text(size:37pt,weight:"bold",title)
  #v(7mm)
  #text(size:13pt,fill:muted,description)
  #v(15mm)
  #line(length:100%,stroke:1pt+gold)
  #v(6mm)
  #grid(columns:(1fr,1fr),gutter:20pt,
    card("Implementation",[Rust checkpoint engine, Candle CPU inference, local browser workbench and read-only MCP.]),
    card("Demonstration model",[Qwen2.5-1.5B-Instruct. 1,543,714,304 stored parameters across 338 tensors.]))
  #v(14mm)
  #note("Evidence before claims.",[Experiments deliberately edit checkpoint copies. Measured integrity, numerical and behavioral results do not establish a naturally occurring vulnerability or an infinite reasoning loop.])
  #v(1fr)
  #text(size:9pt,fill:muted)[28 November 2026 / Local demonstration edition]
]
