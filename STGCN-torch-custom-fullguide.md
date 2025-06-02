# SD-STGCN Usage Guide (Torch 1.12.1 + cu113, Scipy ≥ 1.10.1)

## 0. Requirements

According to STGCN-torch, the requirements are

```
numpy~=1.22.1
pandas~=1.4.3
scikit_learn~=1.5.0
scipy~=1.10.1
torch~=2.3.1
tqdm~=4.66.3
```

For my environment, I used Python 3.10.6 and:

```
networkx                      2.8.8
numpy                         1.23.5
pandas                        1.5.2
scikit-learn                  1.1.3
scipy                         1.15.3
torch                         1.12.1+cu113 (and all its related components)
tqdm                          4.64.1
```

The environments previously for SD-STGCN and DSI are seemingly no longer necessary. 

If really needed, please refer to tips 1.md for environment issues, and please remove all tensorflow related lines in any requirements.txt.

## 1. Merge Data for Training

### Steps

Navigate to the dataset directory:

```bash
cd SD-STGCN/dataset/highSchool/code
```

Edit the configuration in `merge_data.py`. For example, if you want to create data for `n_sources=7`, set:

```python
# Configuration (should match your splitting script)
prop_model = 'SIR'
Rzero = 2.5      # simulation R0
beta = 0.25      # beta
gamma = 0        # simulation gamma
ns = 21200       # number of sequences
nf = 16          # number of frames
T = 30           # simulation time steps
nsrc = 7        # << add this line manually
```

Modify the `base_filename` accordingly:

```python
base_filename = f"{prop_model}_nsrc{nsrc}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire"
```

Example output:

```
Processed SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_1.pickle
Processed SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_2.pickle
Processed SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_3.pickle
Processed SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_4.pickle
Processed SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_5.pickle
...
Successfully merged 5 files into:
../data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle
Total sequences: 21200
Total labels: 21200
Verification: Data count matches original specification
```

Check `data/SIR/split` to see what files you can create. Try `nsrc=1`, `7`, or `10` first.

---

## 2. Training & Testing

Navigate back to the project root:

```bash
cd SD-STGCN
```

Then refer to the template below. Description of arguments is after the training and testing templates.

### Training Command Template

**For Windows:**

```bash
python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_anyname --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > runSIR_nsrc7_anyname.log 2>&1
```

**For Linux:**

```bash
nohup python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_anyname --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > runSIR_nsrc7_anyname.log 2>&1 &
```

### Testing Command Template

If you want to test for same model for training, just

* change `main_SIR_T.py` to `main_SIR_T_testonly.py`
* add flag `--train_flag` in the arguments

I will also provide the template for your reference below:

**For Windows:**

```bash
python -u main_SIR_T_testonly.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_anyname --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 --train_flag > runSIR_nsrc7_anyname_testonly.log 2>&1
```

**For Linux:**

```bash
nohup python -u main_SIR_T_testonly.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_anyname --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 --train_flag > runSIR_nsrc7_anyname_testonly.log 2>&1
```

### Argument Explanation

| Argument       | Value or Description            |
| -------------- | ------------------------------- |
| python filename| main_SIR_T.py or main_SIR_T_testonly.py  |
| `--gt`         | `highSchool`                    |
| `--prop_model` | `SIR`                           |
| `--n_node`     | `774`                           |
| `--n_frame`    | `16`                            |
| `--batch_size` | Suggested: `16`, `32` or `64`   |
| `--epoch`      | Suggested: `20` but 10 is enough|
| `--ks`         | `6` (spatial kernel size)       |
| `--kt`         | `3` (temporal kernel size)      |
| `--sconv`      | `gcn` (`cheb` seems not working)|
| `--save`       | `1`                             |
| `--graph`      | Path to edgelist file           |
| `--seq`        | Path to merged `.pickle` file   |
| `--exp_name`   | Unique name for each experiment |
| `--start`      | `1` (one step after outbreak)   |
| `--end`        | `-1` (final step)               |
| `--random`     | `0` (random seed)               |
| `--lr`         | `0.001`                         |
| `--valid`      | `1`                             |
| `--pos_weight` | `5` (for unbalanced classes)    |
| `--train_pct`  | `0.9434`  (keep it)             |
| `--val_pct`    | `0.0189`  (keep it)             |
| `log output`   | Unique name for each log        |

MAKE SURE YOUR OUTPUT .log NAME AND --exp_name ARE UNIQUE TO PREVENT OVERWRITING!

when you find that the train f1-score and val f1-score is decreasing from the best, you can Ctrl+C to terminate the training process and directely run the code for testing. The code will automatically extract the best checkpoint.

### Example Output

```
[Train Step 310] Acc: 0.9735 | Pred: 0=48251, 1=1285 | Label: 0=49088, 1=448
Epoch  4, Step 310: Time 223.85s | Loss 0.651 | Train F1 0.242 | Val F1 0.235
<< Saving model to ./output/models/highSchool/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_01/STGCN_step4_pytorch.pt ...
New best model saved with validation F1: 0.235 at epoch 4
Final model saved with best validation accuracy: 0.235
Loading model from: ./output/models/highSchool/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_01/STGCN_step4_pytorch.pt
Test F1: 0.2175
Test Prec: 0.1372
Test Recall: 0.5389
Saved test result to ./output/test_res/highSchool/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_01/res.pickle
```

### What to Monitor

1. Whether the model is lazy (predicting only one class).
2. Whether loss is actually decreasing.
3. F1-scores for train and validation sets.
4. Final model saved by best validation F1.
5. Test set F1, precision, recall.

---

## 3. Customize the Model

The core model framework is in:

* `base_model.py`: `STGCN_SI_Nodewise` function
* `layers.py`: all associated layers

**Input and Output Format:**

* `x`: `[B, T, N, C]` where `C=3` for one-hot SIR states
* `y`: `[B, N]`, 0 or 1 indicating if a node is a source
* `logits`: `[B, N, 2]`, logits for binary classification

For your reference, the function should have the following structure:

```python
class STGCN_SI_Nodewise(nn.Module):
    def __init__(self, n_frame, Ks, Kt, blocks, sconv, Lk, keep_prob, pos_weight, n_node=774):
        """
        Customizable STGCN model with standardized input/output interface.
        """
        super(STGCN_SI_Nodewise, self).__init__()

        # Build your custom spatio-temporal backbone
        self.backbone = self.build_backbone(Ks, Kt, blocks, sconv, Lk, keep_prob)

        # Output layer (logits and loss computation)
        self.output = OutputLayer_nodewise(in_channels=blocks[-1][-1], n_node=n_node, pos_weight=pos_weight)

    def build_backbone(self, Ks, Kt, blocks, sconv, Lk, keep_prob):
        """
        Customize this method to build the internal temporal-spatial feature extractor.
        It must transform input shape [B, T, N, C] into [B, T, N, C_out].
        """
        layers = nn.ModuleList()
        for i in range(len(blocks)):
            in_channels, out_channels = blocks[i]
            layers.append(
                MyTemporalSpatialBlock(Ks, Kt, in_channels, out_channels, sconv, Lk, keep_prob)
            )
        return layers

    def forward(self, x, labels=None, training=False):
        """
        Forward pass.
        Args:
            x: Input tensor [B, T, N, C]
            labels: Ground truth tensor [B, N] (optional, only used when training=True)
            training: If True, returns (logits, loss); else returns logits only

        Returns:
            logits: [B, N, 2]
            loss (only if training=True): scalar
        """
        for layer in self.backbone:
            x = layer(x, training=training)  # custom layers must support this flag

        if training:
            logits, loss = self.output(x, labels, training=True)
            return logits, loss
        else:
            logits = self.output(x, training=False)
            return logits
```

**Loss Function:**

```python
loss = F.cross_entropy(logits_valid, y_valid, weight=weights)
```

---

## 4. Graph Convolution Kernel Processing

If you want to use the graph structure, it will be loaded in `main_SIR_T.py`:

```python
W = weight_matrix(gfile)

if sconv == 'cheb':
    L = scaled_laplacian(W)
    Lk_np = cheb_poly_approx(L, Ks, n)
elif sconv == 'gcn':
    Lk_np = first_approx(W, n)

def process_Lk(Lk_np: np.ndarray, sconv: str, Ks: int) -> list[torch.Tensor]:
    if sconv == 'cheb':
        n = Lk_np.shape[0]
        Lk_list = [Lk_np[:, k * n : (k + 1) * n] for k in range(Ks)]
    elif sconv == 'gcn':
        Lk_list = [Lk_np]
    return [torch.tensor(L, dtype=torch.float32) for L in Lk_list]

dataset.Lk = process_Lk(Lk_np, sconv, Ks)
```

* GCN: a single graph kernel `[n, n]`
* Chebyshev: multiple graph kernels `[n, Ks * n]`, split per `k`


## 5. Output reference

Go to commands_and_results.md for previous tensorflow runs on various datasets.

Go to SD-STGCN\runSIR_7_00exfin.log and SD-STGCN\runSIR_7_00exfintestonly.log for train-test separated torch runs:

```bash
python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_00exfin --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > runSIR_7_00exfin.log 2>&1
```
for training, 

```bash
python -u main_SIR_T_testonly.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_00exfin --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 --train_flag > runSIR_7_00exfintestonly.log 2>&1
```
for testing.

Go to SD-STGCN\runSIR_7_01.log for a complete torch run:

```bash
python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 64 --epoch 5 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_01 --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > runSIR_7_01.log 2>&1
```

When replicating these results, please change --exp_name and log name!