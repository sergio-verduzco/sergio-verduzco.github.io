---
title: "Migrating from Roam to Logseq with lots of uploaded files"
date: 2023-09-02
usemathjax: true
---

Taking notes is a basic aspect of my job as a researcher and as a student. More than that, for a long time
I have wanted these notes to be a tool for discovering relations between concepts, leading to new insights.
Using a collection of .txt files arranged by date or by subject does not quite cut it, so I've been searching
for the right note-taking tools.

Over the years I've gone from using a MySQL database to mindmaps (mostly [Freeplane](https://docs.freeplane.org/)) to [TiddlyWiki](https://tiddlywiki.com/). When I got to [Evernote](https://evernote.com/) I felt things
were starting to come together, but something was still missing.

Back in 2020 I started using [Roam Research](https://roamresearch.com/) and
my note-taking needs were almost fulfilled. It's just that Roam is subscription software, not open-source, and stores your data remotely. I'm against all that, but Roam treated me well. So well that for 3 years I was
in denial, and continued paying every month.

But behold the goodness of [Logseq](https://logseq.com/)!

Logseq provides all the functionality I need from Roam, and more. It is free, open-source, and
locally stored. The only thing that stopped me was the pain of migrating 3 years worth of notes.
Luckily, [migrating from Roam Research to Logseq](https://hub.logseq.com/getting-started/uQdEHALJo7RWnDLLLP7uux/how-to-switch-from-roam-research-to-logseq/epbNMUYPWBSjxfrog8v2sH) was not terrible. Still, it had a couple
of challenges. A bothersome one was that I had uploaded a large number of
images and PDF files into my Roam graph. When I exported the graph from Roam and imported it to Logseq
all those files were still in some remote server.

Luckily, my Logseq graph is stored as Markdown files, which can be easily manipulated.
I wrote a Python script (in a [Jupyter notebook](https://jupyter.org/)) to download all external 
PDF and image files in a Logseq graph to a local folder, and change the links accordingly in the Markdown files.

The script is [here](https://github.com/sergio-verduzco/lsad/tree/main). Perhaps it will be useful to someone out there.