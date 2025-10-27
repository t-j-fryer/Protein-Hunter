# Protein-Hunter 😈

**BoltzDesign1** is a molecular design tool powered by the Boltz model for designing protein-protein interactions and biomolecular complexes.

> 📄 **Paper**: [Protein Hunter: exploiting structure hallucination
within diffusion for protein design](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v2.full.pdf)  
> 🚀 **Colab**: https://colab.research.google.com/drive/1JBP7iMPLKiJrhjUlFfi0ShQcu8vHn2gI#scrollTo=CzE1iBF-ZCI0

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yehlincho/Protein-Hunter.git
   cd Protein-Hunter
   ```

2. **Run the automated setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

> ⚠️ **Note**: AlphaFold3 setup is not included. Please install it separately by following the [official instructions](https://github.com/google-deepmind/alphafold3).

The setup script will automatically:
- ✅ Create a conda environment with Python 3.10
- ✅ Install all required dependencies
- ✅ Set up a Jupyter kernel for notebooks
- ✅ Download Boltz model weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta
- ❌ AF3 must be installed separately

---

We have implemented two different AF3-style models in our Protein Hunter Pipeline (more models will be added in the future):
- Boltz1/2
- Chai1