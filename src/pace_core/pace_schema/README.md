# schema/ · PACE JSON Schema(机读事实源)

- 体例:JSON Schema **draft-07**;`$id` 形如
  `pace/<name>.schema.json`;上游 registry 版本为 `pace-0.2`；SceneDoc 的 `schemaVersion`
  单独由 `types_v1.py` 中的 `SCHEMA_VERSION` 定义（当前 `pace-1.3`）。
- **additionalProperties 策略**:对象默认 `true`(向前兼容,读取方容忍新增字段);
  受控枚举字段严格 enum(取值即 `vocab/` 对应词条的 value)。
- dormant 字段(见 registry)**同样在 schema 中定义**——写入合法但 lint 给 warning。
- 文件:
  - `project_manifest.schema.json` — Project Manifest(项目级根:标题 / 流程状态 / 项目级 artifacts[])
  - `asset_index.schema.json` — 实体资产台账(`entities/characters.json` / `props.json` / `locations.json` / `voices.json`)
    的通用 schema;实体资产引用只保存 `assets://` 到 `uri`
  - `scene_manifest.schema.json` — Scene Manifest(场级根:shotDefaults / physicalLayout / narrativeMeta;不再内嵌 shots[])
  - `pillar.setup.schema.json` / `pillar.camera.schema.json` / `pillar.lighting.schema.json` / `pillar.events.schema.json`
  - `semantics.schema.json` — OMC-lite(version/task/participant/context/relationship)
  - `shot_manifest.schema.json` — Shot Manifest(shot 级根:四柱 pace / continuity / audio / review / shot 级 artifacts[])
  - `panel_manifest.schema.json` — Panel Manifest(panel 级根:关键帧元数据 / panel 级 artifacts[])
- 版本一致性检查：[test_schema_version.py](../../../tests/test_schema_version.py)。
  该测试检查版本常量和 schema 声明，不替代对业务输入的完整校验。
