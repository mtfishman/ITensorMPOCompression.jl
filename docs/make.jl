using Documenter, ITensorMPOCompression, ITensors

makedocs(;
  sitename="ITensorMPOCompression",
  format=Documenter.HTML(; prettyurls=false),
  modules=[ITensorMPOCompression],
)
