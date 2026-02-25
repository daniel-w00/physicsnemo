                                       
  What we implemented                                                                                                                                                                                                                                                                                               
                                                            
  datasets/cwb.py                                                                                                                                                                                                                                                                                                   
  - Added embedding_path=None parameter to ZarrDataset.__init__ and get_zarr_dataset()                                                                                                                                                                                                                              
  - On init: loads alpha_earth array from zarr, fills NaN→0, crops to 448×448, stores as torch tensor
  - __getitem__: concatenates the 64 embedding channels onto the ERA5 input after reshape_fields
  - input_channels(): appends 64 alpha_earth_0 … alpha_earth_63 channel metadata entries

  conf/base/dataset/cwb.yaml — added embedding_path: null

  conf/config_training_taiwan_regression-alex.yaml — set embedding_path to the zarr path

  conf/config_generate_taiwan.yaml — same

  Result

  Model input goes from 12+4=16 → 12+64+4=80 channels. train.py picks this up automatically.

  ---
  What still needs to be done / verified

  1. Check the zarr key — we assumed the array is stored as emb_zarr["alpha_earth"]. Run zarr.open(...).tree() to confirm, and fix the key in cwb.py if different.
  2. Unit test — instantiate the dataset and assert input.shape == (76, 448, 448).
  3. Training dry run — confirm the model initializes without shape errors.