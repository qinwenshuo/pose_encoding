import pandas as pd


def classify_model_family(uid: str) -> str:
    u = uid.lower()
    checks = [
        # Training-method families (check before generic architecture names)
        (lambda u: u.startswith(("clip_", "slip_")),                                    "CLIP"),
        (lambda u: u.startswith("dino_"),                                                "DINO"),
        (lambda u: u.startswith("vissl_"),                                               "VISSL"),
        (lambda u: u.startswith("vicreg_"),                                              "VICReg"),
        (lambda u: u.startswith("bit_expert"),                                           "BiT"),
        # Architecture families
        (lambda u: "beitv2" in u,                                                        "BEiT/BEiTv2"),
        (lambda u: "beit" in u,                                                          "BEiT/BEiTv2"),
        (lambda u: "deit3" in u,                                                         "DeiT/DeiT3"),
        (lambda u: "deit" in u,                                                          "DeiT/DeiT3"),
        (lambda u: "convnext" in u,                                                      "ConvNext"),
        (lambda u: "swin" in u,                                                          "Swin"),
        (lambda u: any(x in u for x in ("mixer_", "gmlp", "gmixer", "convmixer")),      "MLP-Mixer/gMLP"),
        (lambda u: any(x in u for x in ("cait_", "convit", "levit",
                                         "nest_", "mvitv2")),                            "ViT/CaiT/ConViT"),
        (lambda u: any(x in u for x in ("nfnet", "efficientnet", "efficientformer")),   "EfficientNet/NfNet"),
        (lambda u: any(x in u for x in ("mobilenet", "mobilevit", "mnasnet",
                                         "lcnet", "hardcorenas", "mixnet",
                                         "ghostnet", "shufflenet")),                    "MobileNet/Lightweight"),
        (lambda u: "densenet" in u,                                                      "DenseNet"),
        (lambda u: "regnet" in u,                                                        "DLA/RegNet"),
        (lambda u: any(x in u for x in ("dla_", "dla26", "dla34", "dla46",
                                         "dla60", "dla102", "dla169")),                 "DLA/RegNet"),
        (lambda u: any(x in u for x in ("seresnet", "seresnext", "senet",
                                         "ecaresnet", "eca_resnext",
                                         "ig_resnext", "bat_resnext",
                                         "lambda_resnet")),                             "ResNet/ResNext/SE"),
        (lambda u: "resnet" in u or "resnext" in u,                                     "ResNet/ResNext/SE"),
        (lambda u: u.startswith(("x3d_", "slowfast_", "slow_r", "i3d_", "c2d_"))
             or "timesformer" in u,                                                      "Video"),
        (lambda u: any(x in u for x in ("inception", "darknet", "cs3", "csedarknet",
                                         "squeezenet", "alexnet", "dpn", "gernet",
                                         "edgenext", "botnet", "halonet", "haloreg",
                                         "lamhalo", "ssd")),                            "Other CNNs"),
    ]
    for test, family in checks:
        if test(u):
            return family
    return "Other"


df = pd.read_csv("experiments/all_models_list.csv")
df["model_family"] = df["Model UID"].apply(classify_model_family)

out_path = "data/processed/grouped_models.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} models to {out_path}")
print(f"\nFamily counts:")
print(df["model_family"].value_counts().to_string())
print(f"\nOther (unclassified):")
other = df[df["model_family"] == "Other"]["Model UID"].tolist()
for uid in other:
    print(f"  {uid}")
