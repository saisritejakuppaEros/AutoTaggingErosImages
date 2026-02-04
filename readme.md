to activate conda envs
```
conda activate rnd_autotag_v2
```


To run segmetntation on source images
```
python main.py --config config.toml segment-source
python main.py --config config.toml segment-target --folder 00291
```