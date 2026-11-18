# Final submission checklist

INFR11269 coursework. Deadline: **Fri 20 November 2026, at the time shown on Learn**.

This list is based on `docs/01`–`05` and the three course announcements:
- task-guide questions and submission materials;
- the two video questions;
- video submission through Media Hopper Create.

Where it says "✅ 18 Nov", the item was checked in this repository on Wed 18 Nov. "⚠️" means it is still to do.

## 0. What you hand in

| # | Item | File | How it is handed in | Status (18 Nov) |
|---|---|---|---|---|
| 1 | Group report: 70% of the mark, with the video | `report.pdf` | Attached to the Learn submission. **Both members upload the same PDF**: Learn picks one of you at random for marking and review | ⚠️ The group information is still the template placeholder, and the AI declaration doesn't mention the video (section 1) |
| 2 | Supporting video: ≤ 10 min, no separate mark | `video.mp4` | Uploaded to **Media Hopper Create**, then embedded in the Learn submission text box. Do not upload the file to Learn | Recorded. Check it in section 2 |
| 3 | Supplementary code: optional | `supplementary_code.zip` | Attached to the Learn submission | ✅ Built: `submission/supplementary_code.zip` (650 KB, 41 files) |
| 4 | Submission receipt | — | Saved from Learn after you click Submit | — |

Separately, after the deadline, each of you writes the **individual review** (30%; see section 7).

Suggested order:
- **Wed 18 Nov:** fix the report and check the video file.
- **Thu 19 Nov:** upload the video to Media Hopper Create (it needs processing time) and submit on Learn.
- **Fri 20 Nov:** keep it free as a buffer.

Check on Learn whether a second attempt is allowed before relying on one.

---

## 1. Report: `report.pdf`

### Fix first
- [ ] ⚠️ **Group information** (`report/v2/report.tex`, lines 446–460): replace `Group ID: XXX`, and the two `John Doe` and `sXXXXXXX` entries, with your Group ID, names and UUNs. That page is the only place they may appear.
- [ ] ⚠️ **AI declaration** (lines 437–441): add this bullet after the "Report" bullet, before `\end{itemize}`. It also covers this checklist and the zip's README:

  ```latex
  \item Video and submission: it drafted the narration script and the slides for the video, suggested the commands for the live demonstration (including a small shell function that formats the demo output), prepared a submission checklist and wrote the README and packaging script for the supplementary code. We checked the content, ran the demonstration and recorded the narration ourselves.
  ```
- [ ] Make the same two edits in the Overleaf project, or upload the edited `report.tex`. Recompile, download the PDF and name it `report.pdf`. One of you compiles it, and you both upload that same file.

### Check the downloaded PDF
Run these from the folder that holds `report.pdf`. `pdfinfo`, `pdftotext` and `ffprobe` are already installed on this Mac via Homebrew. The commands were tested on the current PDF.

```bash
N=$(pdfinfo report.pdf | awk '/^Pages:/{print $2}'); echo "pages: $N"
```

- [ ] **Main body ≤ 6 pages:** the References heading must be on page 6 or earlier. ✅ 18 Nov: page 6 of 10.

  ```bash
  for i in $(seq 1 "$N"); do pdftotext -f "$i" -l "$i" report.pdf - | grep -q '^References$' && echo "References on page $i of $N"; done
  ```
- [ ] **One final page:** the AI declaration and group information share the last page, after the appendix, with no technical content. That page is removed before peer review. ✅ 18 Nov: page 10 of 10. Check again after your edits that nothing spills onto an extra page:

  ```bash
  pdftotext -f "$N" -l "$N" report.pdf - | grep -v -E '^[[:space:]]*[0-9]*[[:space:]]*$' | head -30
  ```
- [ ] **Nothing identifying before the last page.** Put your surnames and Group ID in place of the capitals; no output means it's clean. ✅ 18 Nov: nothing found on pages 1–9.

  ```bash
  pdftotext -f 1 -l $((N-1)) report.pdf - | grep -n -i -E 'SURNAME1|SURNAME2|GROUPID|s[0-9]{7}|@ed\.ac\.uk'
  ```
- [ ] **Neutral metadata:** no Title or Author with your names. ✅ 18 Nov: there is no Title or Author field; the Creator is "LaTeX with hyperref".

  ```bash
  pdfinfo report.pdf | grep -E '^(Title|Author|Subject|Keywords|Creator)'
  ```
- [ ] **Format checker:** no FAIL; look at any WARN in the PDF. ✅ 18 Nov on the current PDF: `Status: PASS`. It found 10.0 pt body font, the header on 10/10 pages, the review line numbers, References on page 6 and an estimated main body of 6 pages (limit 6). The system Python on this Mac has no PyMuPDF, so use a throwaway environment:

  ```bash
  python3 -m venv /tmp/fmtcheck && /tmp/fmtcheck/bin/pip install -q -r report/format_checker/requirements.txt
  /tmp/fmtcheck/bin/python report/format_checker/check_report_format.py /path/to/report.pdf
  ```
- [x] ✅ **Template untouched**:
  - `\author{Anonymous coursework submission}` and the `\lhead{...}` line are unchanged;
  - `\iclrfinalcopy` and the abstract are still commented out;
  - References come before the appendix;
  - only the title and three plotting packages differ from `report/latex_template/report.tex`.
- [ ] Open the PDF and look at every page: figures render, and nothing runs into the margin.

### Content the course asks for (already in the report)
- [x] ✅ Output of `make test` and `./run.sh --profile`, from one node, and the GPU name:
  - the key numbers are in §3, §4.1 and Table 1;
  - the full output is in Appendix A and Table 4.

  Announcement 1 allows this split.
- [x] ✅ One answer from each of the four task guides, in §4.2 ("Answers to the task-guide questions"). The Introduction points to it, as announcement 1 recommends.
- [x] ✅ RAG investigation:
  - question (§1);
  - baseline and change (§2.2);
  - controls and metrics (§3);
  - quality and cost (§4.3–4.4, Figure 2, Tables 2–3);
  - why (§4.5);
  - limitations (§5).
- [x] ✅ Supplied and own work are kept apart (Figure 1: blue is ours).

---

## 2. Video: `video.mp4`

### Check the file before uploading
These commands were tested with `ffprobe`/`ffmpeg` on this Mac.

- [ ] The file is named `video.mp4`.
- [ ] **≤ 10:00.** This prints the length as m:ss:

  ```bash
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 video.mp4 | awk '{printf "%d:%02d\n", $1/60, $1%60}'
  ```
- [ ] **MP4 with H.264 and AAC.** Expect `codec_name=h264|codec_type=video` and `codec_name=aac|codec_type=audio`:

  ```bash
  ffprobe -v error -show_entries stream=codec_type,codec_name -of compact=p=0 video.mp4
  ```
- [ ] **No identifying metadata.** No `title`, `artist`, `author`, `comment` or `com.apple.quicktime.*` tag should hold a name; `encoder`, `handler_name` and `major_brand` are fine:

  ```bash
  ffprobe -v error -show_entries format_tags:stream_tags -of default=nw=1 video.mp4
  ```

  If you find any, strip all metadata without re-encoding, then check again and rename the clean copy to `video.mp4`:

  ```bash
  ffmpeg -i video.mp4 -map_metadata -1 -c copy video_clean.mp4
  ```
- [ ] If the file is over the size limit shown on Learn or Media Hopper Create, compress it with HandBrake (Fast 1080p30) as described in `docs/05_video_production_guide.pdf`. Keep the original.

### Watch the whole exported file once more
- [ ] **Part A** (≤ 8 min) explains the system, your work and the findings. It shows an actual run: the command and configuration, the input, the output, and what the output shows.
- [ ] Every cut inside a run is marked "Waiting time shortened".
- [ ] **The final two minutes** hold both answers, each introduced by name and about a minute long, ending by 10:00:
  - "Question 1": your main design choice and why;
  - "Question 2": how you checked the work, with an example.
- [ ] The narration is audible throughout; terminal text and figures are readable at full screen.
- [ ] **Nothing identifying on screen:**
  - no names, UUN (shell prompt, ssh login line, `/home/...` paths, window titles), Group ID or email;
  - no browser profile or notifications;
  - no face and no ELM key.

### Upload to Media Hopper Create
- [ ] Sign in to Media Hopper Create with your University account. **+ Create → Upload**, choose `video.mp4`, then **Save**.
- [ ] Use a neutral video title and display name, as the course team asks (for example "Coursework video"). Leave the description and tags empty.
- [ ] Wait for processing to finish, then play the video in Media Hopper Create from start to end. Check the sound and that the text is readable.

### Embed it in the Learn submission
- [ ] Open the coursework assignment in Learn and go to the submission area.
- [ ] Click inside the submission text box, then **+ → Content Market → Media Hopper Create Embed (Ultra)**.
- [ ] Select your video and follow the prompts to submit it as an assignment and insert it into the submission.
- [ ] Check that the embedded player appears and plays. If a step looks different, see the University guide "Submitting Media to a Learn Assignment".
- [ ] Uploading to Media Hopper Create on its own does **not** submit anything.

---

## 3. Supplementary code: `supplementary_code.zip`

✅ 18 Nov: `bash submission/build_supplementary_zip.sh` built `submission/supplementary_code.zip`.
- Before writing the zip, the script checks every file for your git user name, UUN-like strings, `/home/` and `/Users/` paths, `@ed.ac.uk` addresses and API keys. It stops if anything matches.
- Rebuild the same way if you change any file it includes.
- Build it on this Mac: `rag/extensions/paired_analysis.py` is only here until you commit and push it.

Inside (`supplementary_code/`, with the same paths as the course repository):
- `README.md`, from `submission/README_supplementary.md`, for the markers and the peer reviewer. It lists the files, dependencies and commands, and marks which files are ours.
- The five kernel files and the HW1 outputs (`hw1-asr/hw1_*.txt`). In `hw1_profile.txt`, the library path is shortened to `hw1-asr/build/...`, because the original contains your UUN.
- RAG code:
  - new: `reranker.py`, `bench_rerank.py`, `paired_analysis.py`, `evaluate_rerank.py`, `test_rerank.py`, `cross.py`;
  - changed: `rag.py`, `answer_judge.py`, `generate_answers.py`.
- `changes_to_supplied_files.patch`: the exact diff of the 8 supplied files you changed, against the supplied commit `6bbdfbc`. It was checked to apply cleanly to the supplied code.
- The result files behind the report's tables and figures. `paired_analysis.py` reproduces the confidence intervals from them.

Left out on purpose:
- `CLAUDE.md`, `run.txt`, `notes.txt` and `rag_notes.txt`, which contain your UUN or personal notes;
- `docs/`, `report/`, `video/`, `rag/.env`;
- models, corpus data and caches (`judge_cache.jsonl`, `rag/.cache/`), `build/`, the smoke-test run;
- Git history;
- supplied code you did not change (announcement 1).

- [ ] Check that your teammate's surname and the Group ID are not in the zip either:

  ```bash
  EXTRA_TERMS="teammate_surname groupid" bash submission/build_supplementary_zip.sh
  ```

  Words shorter than 3 letters are skipped. If your Group ID is only digits, leave it out, because it would match numbers in the result files. If the check fails, the script deletes the old zip, so a failed check never leaves a stale zip behind.
- [ ] Skim the zip's README once:

  ```bash
  unzip -p submission/supplementary_code.zip supplementary_code/README.md
  ```

---

## 4. Submit on Learn (each member)

Learn randomly picks one member's submission for marking and review, so make **both submissions complete and identical**.
- [ ] Both: the same `report.pdf`.
- [ ] Both: the same `supplementary_code.zip`.
- [ ] Both: the group's one video embedded. The second member can upload the same `video.mp4` to their own Media Hopper Create and embed it. If you're unsure whether the second member must embed it too, ask the course team before the deadline.
- [ ] Attach `report.pdf` and `supplementary_code.zip` as instructed on Learn, check the video is embedded, and click **Submit**.
- [ ] Look for the success confirmation and **save the submission receipt**.
- [ ] Open the submission again:
  - download `report.pdf` and check it is the right version, with all pages;
  - check the zip opens;
  - check the video plays.

---

## 5. Last look before you click Submit
From `docs/01_submission_requirements.pdf`, page 3:
- [ ] The PDF uses the supplied template and its main body is at most six pages.
- [ ] All technical content comes before the single final declaration and student-information page.
- [ ] The report, video and code follow the anonymity rules: names, UUNs and the Group ID appear only on the report's last page.
- [ ] The video is at most ten minutes, contains a system demonstration and keeps its final two minutes for both questions.
- [ ] The files open correctly and everything you intended is in the Learn submission.

---

## 6. Housekeeping (not handed in)
- [ ] Keep your GitHub copy of this repository **private**: it holds your solutions, and `CLAUDE.md` and `run.txt` contain your UUN.
- [x] ✅ `rag/.env` has never been committed. Keep it that way, and never share it.
- [x] ✅ Only the five task files differ from the supplied `hw1-asr/` code. `ops.py`, `model.py`, `benchmark.py`, `provided.cu` and the tests are unchanged, and no cuBLAS, cuDNN, CUTLASS, Thrust or PyTorch calls appear in the kernels.
- [ ] Optional: commit `submission/` and `rag/extensions/paired_analysis.py`. The zip itself is git-ignored.

---

## 7. After the deadline: individual review (30%, each student on their own)
You each receive one anonymised report, without its last page, plus any code; you do not get the video. Fill in four fields in Learn:

| Field | Max words | What goes in it |
|---|---|---|
| Summary | 150 | the question, approach and findings, in your own words |
| Strengths | 300 | at least 3, numbered, each explained with reference to the work |
| Weaknesses | 300 | at least 3, numbered, each saying why it matters |
| Detailed Comments | 300 | one or two points developed into specific suggestions |

Keep your name and UUN out of the review text. See `docs/03_individual_review_requirements.pdf` and `docs/04_individual_review_rubric.pdf`.
