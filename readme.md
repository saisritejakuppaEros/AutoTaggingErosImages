Pipeline

Make sure you have the csv file , which has all the tags in the first place, seperated by '-'
Edit the config files.

Embeddings generation for source is

```
# Step 1: Segment source images
python main.py --config config.toml segment-source

# Step 2: Generate embeddings for source segments
python main.py --config config.toml embed-source

# Step 3: Build FAISS index from source embeddings
python main.py --config config.toml build-faiss
```


Target tagging generation is
```
# For folder 00290
python main.py --config config.toml segment-target --folder 00290
python main.py --config config.toml embed-target --folder 00290
python main.py --config config.toml tag-target --folder 00290
```



To viz the tags for a sample folder please run
```
streamlit run utils/visualize_tags.py
```