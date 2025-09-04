1. # Knowledge Graph Deployment

All ToxKG data are stored in a neo4j.dump file. To deploy it locally, follow the steps below (note: importing a dump file requires using the same major version of Neo4j as our build, i.e., Neo4j Server 5.x).

```python
neo4j stop
neo4j-admin database load neo4j --from=/path/neo4j.dump --force
neo4j start
```

# 2. Environment Setup

The environment used in this project has been fully encapsulated in the `environment.yml` file. You can install it with a single command as follows:

```python
conda env create -f environment.yml -n myenv
conda activate myenv
```

# 3. Code Usage Instructions

## 3.1 File Description

### **data** directory

Location for storing data and input features.

#### **data/KG/r-gcn** directory

Contains data files related to the R-GCN model.

```python
common_cids.csv     --Information containing the CIDs, SMILES representations, and the 12 toxicity prediction results of the common molecules shared by Tox21 and the knowledge graph
gnn_input_fp.csv    --Information containing CIDs and five different molecular fingerprints
filtered_triples.csv    --Complete triplet information of molecules with shared CIDs
gnn_input_genes.csv     --Gene list file
gnn_input_pathways.csv  --File containing the pathway list
compound_master.csv --Contains CID, SMILES, {fingerprint_cols…}, and {12 toxicity columns}
```

### GAT/GCN/GPS/HGT/HRAN/R-GCN

Directory containing the training code and training results for the six GNN models.

#### **save** folder

Stores the training results.

#### **train** folder

Contains the training code.

#### **data_utils**

Contains data preprocessing scripts.

# 4. Model Training Commands

## 4.1 GCN Execution Command

```python
python gcn2-bac.py --epochs 1000 --hidden 512 --heads 4
```

## 4.2 GAT Execution Command

```python
python gat2-bac.py --epochs 1000 --hidden 512 --heads 4
```

## 4.3 GPS Execution Command

```
python train_gps_gine_cv.py --epochs 1000 --hidden 512 --heads 4
```

## 4.4 HGT Execution Command

```python
python train_hgt_cv.py --epochs 1000 --hidden 128 --lr 5e-4 --embed_dim 32
```

## 4.5 HRAN Execution Command

```python
python train_hran_pos_weight_cv.py \
       --gene_col geneSymbol --pathway_col pathwayId \
       --epochs 1000 --hidden 512
```

## 4.6 R-GCN Execution Command

```python
python train_rgcn_cv2.py \
       --gene_col geneSymbol --pathway_col pathwayId \
       --epochs 1000 --hidden 512
```

