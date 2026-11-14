# AstraGuard: AI-Powered Satellite Intelligence & Disaster Response Platform 🛰️

**Department of Electronics & Communication Engineering**  
*Maulana Azad National Institute of Technology (MANIT), Bhopal*  
**Under the Supervision of:** Dr. Lalita Gupta Ma'am  

**Team Members:**
- Sujal Sharma (2311401108)
- Aditya Tomar (2311401139)
- Mohammad Maasir (2311401143)
- Manav Marimat (2311401159)
- Nikhil Rajpoot (2311401120)

---

## 📌 Project Overview
AstraGuard is a modular AI-driven satellite intelligence and disaster analysis platform. This repository contains the training and evaluation implementations for **Model 3 (Building / Urban Segmentation)** and **Model 4 (Temporal Change Detection)** using SpaceNet 7 satellite imagery.

---

## 🏗️ Model 3: Building & Urban Footprint Segmentation
- **Dataset:** SpaceNet 7 (High-resolution multi-band satellite imagery)
- **Architectures Compared:** U-Net (ResNet-34) vs. DeepLabV3+ (ResNet-50)
- **Loss Function:** Dice + Binary Cross Entropy (BCE)
- **Results:**
  | Architecture | Backbone | Best IoU | Best F1 |
  |---|---|:---:|:---:|
  | **U-Net** | ResNet-34 | `0.2007` | `0.3339` |
  | **DeepLabV3+** | ResNet-50 | **`0.2425`** | **`0.3900`** |

Notebook: [`Model_3_Building_Segmentation.ipynb`](./Model_3_Building_Segmentation.ipynb)

---

## 🔄 Model 4: Temporal Change Detection
- **Dataset:** SpaceNet 7 Bi-temporal monthly image pairs
- **Task:** Predict pixel-wise binary change masks (urban development / building appearances)
- **Architectures Compared:** Siamese U-Net vs. ChangeFormer (Hierarchical Transformer)
- **Loss Function:** Combined Dice + Focal Loss (to address extreme class imbalance)
- **Results:**
  | Architecture | Best IoU | Best F1 |
  |---|:---:|:---:|
  | **Siamese U-Net** | **`0.4375`** | **`0.4375`** |
  | **ChangeFormer** | **`0.4375`** | **`0.4375`** |

Notebook: [`Model_4_Change_Detection.ipynb`](./Model_4_Change_Detection.ipynb)

---

## 🚀 How to Run on Google Colab
1. Open either notebook (`.ipynb`) in this repository.
2. Replace `github.com` in the URL with `colab.research.google.com/github/` to launch directly in Colab, or download and upload to your Google Drive.
3. In Google Colab, select: **Runtime → Change runtime type → T4 GPU**.
4. Run all cells sequentially.
