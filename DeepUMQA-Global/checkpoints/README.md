请把 DeepUMQA-Global 的权重文件放在这个目录下，或者通过 `--ckpt-path` 指向其他 checkpoint 文件或目录。

当前仓库默认示例使用：

`val_best-epoch=26-val_loss=0.00057.ckpt`

程序会自动扫描目录中的 `*.ckpt`、`*.pt` 和 `*.pth`。
