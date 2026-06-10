## 修改内容

- 

## 为什么修改

- 

## 测试结果

请填写实际运行过的命令和结果，例如：

```text
pytest -q
```

## 影响范围

- [ ] Stage 1 mock pipeline
- [ ] Stage 2 MuJoCo executor
- [ ] Stage 3 YOLO/BODex input interfaces
- [ ] cuRobo planning
- [ ] MuJoCo simulation
- [ ] docs only
- [ ] other:

## 是否修改文档

- [ ] 是
- [ ] 否

说明：

## 是否涉及大文件/数据/模型权重

- [ ] 否
- [ ] 是，已说明处理方式：

## Reviewer 检查清单

- [ ] 修改目标清楚，范围合理
- [ ] 没有直接提交 `outputs/`、`__pycache__/`、`.pytest_cache/`
- [ ] 没有提交模型权重、数据集或不必要的大文件
- [ ] 测试结果可信，没有把失败写成成功
- [ ] 文档和命令与当前代码一致
- [ ] 若涉及 YOLO/BODex，仍保持它们作为上游模块，不在本仓库做深度改造
- [ ] 若涉及 cuRobo/MuJoCo，说明了 demo 资源和真实机械臂资源的边界
