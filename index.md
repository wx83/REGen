## Contents:

1) [Section 1: Summary of the Compared Models](#section-1-summary-of-the-compared-models)

2) [Section 2: Dataset Annotation](#section-2-dataset-annotation)

3) [Section 3: Qualitative Examples](#section-3-qualitative-examples)

4) [Section 4: More Examples](#section-4-more-examples)

5) [Section 5: Zero-shot Examples](#section-5-zero-shot-examples)



---

# Section 1: Summary of the Compared Models

| Model  | Narration Type | Script Generation | Retriever  |
|-------------|-------------|-------------|-------------|
| A2Summ | Extraction | -- | -- |
| Extraction-then-smoothing(ETS) | Extraction | -- | --|
| TeaserGen | Abstraction | -- | -- |
| GPT-4o-DQ | Extraction & Abstraction |  Direct Quote | -- |
| GPT-4o-SP-DQ | Extraction & Abstraction |  Direct Quote(with speaker annotation)| -- |
| GPT-4o-SP-TV |  Extraction & Abstraction | Indirect Quote(with speaker annotation) |  QuoteRetriever-TV | 
| GPT-4o-DQ |  Extraction & Abstraction | Direct Quote | -- |
| REGen-DQ |  Extraction & Abstraction | Direct Quote | -- | 
| REGen-IDQ-T |  Extraction & Abstraction | Indirect Quote | QuoteRetriever-T |
| REGen-IDQ-TV |  Extraction & Abstraction | Indirect Quote | QuoteRetriever-TV |

# Section 2: Dataset Annotation
> We annotate the start time, end time, segment type (speaker or quotable interview), and transcribed text for both teasers and documentaries as follows:

<figure style="text-align: center;">
  <img 
    src="./data_annotation.jpg" 
    alt="Dataset annotation diagram showing Teaser vs. Main Documentary" 
    style="width: 1400px; height: auto;"
  >
  <figcaption><strong>Dataset annotation for Teaser and Main Documentary</strong></figcaption>
</figure>

---

# Section 3: Qualitative Examples
<p style="text-align: center;">
  <strong>Video Title: documenta 14 - learning from Athens | DW Documentary</strong>
</p>
<figure style="text-align: center;">
  <img 
    src="./ag.jpg" 
    alt="Qualitative Example" 
    width="1000"
  >
  <figcaption><strong>Visualization</strong></figcaption>
</figure>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-T</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/agij_IxGjCI?start=107" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/llama_quote_mt_only.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
        <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<p style="text-align: center;">
  <strong>Video Title: Apocalypse (Full Episode) | The Story of God with Morgan Freeman</strong>
</p>
<figure style="text-align: center;">
  <img 
    src="./at.jpg" 
    alt="Qualitative Example" 
    width="1000"
  >
  <figcaption><strong>Visualization</strong></figcaption>
</figure>


<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-T</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/ATvKJ_HftNs?start=118" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/ATvKJ_HftNs/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/ATvKJ_HftNs/llama_quote_mt.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
        <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/ATvKJ_HftNs/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

--- 

# Section 4: More Examples
> We compare our model performance with TeaserGen, A2Summ and GPT-based models in the following section

<p style="text-align: center;">
  <strong>Video Title: Is Parkinson's disease related to pesticide use?</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/6i2sJwxw5Uc/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/6i2sJwxw5Uc/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/6i2sJwxw5Uc/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/6i2sJwxw5Uc/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/6i2sJwxw5Uc/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---
<p style="text-align: center;">
  <strong>Video Title: documenta 14 - learning from Athens</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
        <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/agij_IxGjCI/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: The dirty business of beauty</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/eAIKvD_gLJo/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/eAIKvD_gLJo/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/eAIKvD_gLJo/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/eAIKvD_gLJo/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/eAIKvD_gLJo/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: Lost at Sea (Full Episode)Extreme Rescues</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/khyuH_QfoWU/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/khyuH_QfoWU/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/khyuH_QfoWU/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/khyuH_QfoWU/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/khyuH_QfoWU/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: Saving kids from the Mafia in Italy</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/noo8-_LYIpA/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/noo8-_LYIpA/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/noo8-_LYIpA/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/noo8-_LYIpA/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/noo8-_LYIpA/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: "Doctors, apps and artificial intelligence - The future of medicine</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/vyit-1zKsZ4/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/vyit-1zKsZ4/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/vyit-1zKsZ4/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/vyit-1zKsZ4/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/vyit-1zKsZ4/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: The apostle comes from Africa — a contemporary passion story</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/gYo-icA2848/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/gYo-icA2848/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/gYo-icA2848/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/gYo-icA2848/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/gYo-icA2848/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<p style="text-align: center;">
  <strong>Video Title: Love & marriage in Egypt and Taiwan – Whose choice is it?</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/UNXDA-h6pP8/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/UNXDA-h6pP8/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/UNXDA-h6pP8/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/UNXDA-h6pP8/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/UNXDA-h6pP8/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: Archeology – exploring the past with modern technology</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/VpK8fpqPJT0/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/VpK8fpqPJT0/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/VpK8fpqPJT0/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/VpK8fpqPJT0/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/VpK8fpqPJT0/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Video Title: Beyond Death (Full Episode) The Story of God with Morgan Freeman</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
    <td style="text-align: center;"><strong>A2Summ</strong></td>
    <td style="text-align: center;"><strong>GPT-4o-SP-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/wZORPVmXN7k/teasergen.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/wZORPVmXN7k/a2summ.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/wZORPVmXN7k/gpt_closest_teaser.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>REGen-DQ</strong></td>
  </tr>
  <tr>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/wZORPVmXN7k/llama_quote_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/wZORPVmXN7k/llama_contents_nn.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

# Section 5: Zero-shot Examples
> Zero-shot Examples on Lecture Video and News Videos Teaser Generation Task.


### News Videos

<p style="text-align: center;">
  <strong>NBC Nightly News Full Episode</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/AOAmc8nW1OA?start=80" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/news_videos/AOAmc8nW1OA/news_video_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/news_videos/AOAmc8nW1OA/teasergen_news.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>NBC Nightly News Full Episode</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/KVbknpFS-Hw/?start=90" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/Subjective/versionA/news_videos/KVbknpFS-Hw/news_video_mv.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/Subjective/versionA/news_videos/KVbknpFS-Hw/teasergen_news.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

### Lecture Videos

<p style="text-align: center;">
  <strong>Machine Learning: Coordinated Representations (Multimodal Machine Learning)</strong>
</p>
<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/2_dZ5GBlRgU" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/lecture_videos/2_dZ5GBlRgU/teasergen_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionA/lecture_videos/2_dZ5GBlRgU/llama_quote_mv_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Psychology: PSY101 Conditioning and Learning</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/j2-9yymHfeU" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://wx83.github.io/REGen/versionB/lecture_videos/j2-9yymHfeU/llama_quote_mv_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionB/lecture_videos/j2-9yymHfeU/teasergen_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<p style="text-align: center;">
  <strong>Biology: CurrentTopicsLecture3Ch3</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/DZSEErNZ1d4" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/DZSEErNZ1d4/llama_quote_mv_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
      <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/DZSEErNZ1d4/teasergen_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<p style="text-align: center;">
  <strong>Biology: Endocrine System - Pituitary Gland THE MASTER GLAND</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/DM4MWxnXOzM" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/DM4MWxnXOzM/teasergen_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/DM4MWxnXOzM/llama_quote_mv_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<p style="text-align: center;">
  <strong>Dentistry: Oral Medicine | ASA Classification | INBDE</strong>
</p>

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>REGen-IDQ-TV</strong></td>
    <td style="text-align: center;"><strong>TeaserGen</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/-8QB3Rbeqoc" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/-8QB3Rbeqoc/llama_quote_mv_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/NeurIPS/versionA/lecture_videos/-8QB3Rbeqoc/teasergen_lecture.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---
