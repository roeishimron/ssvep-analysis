FROM roeish/agentic-python-analysis:3.14

# LyX + XeLaTeX toolchain for compiling the Hebrew seminarion.lyx
# (uses non-TeX fonts -> XeLaTeX; David CLM / Hadasim CLM -> fonts-culmus;
#  pgfplots/tikz -> texlive-pictures; citations.bib -> biber + bibtex-extra;
#  Hebrew babel/polyglossia -> texlive-lang-other)
COPY apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends `cat /tmp/apt-requirements.txt` \
 && rm -rf /var/lib/apt/lists/* /tmp/apt-requirements.txt
