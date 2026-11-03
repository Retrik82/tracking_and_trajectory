"""
Минимальное чтение бинарного FBX (7.x): узлы, свойства, сетки и их трансформации.

Нужен, чтобы взять из 3D-модели грузовика габариты, положение опорной точки (pivot)
и части (тягач, прицеп, колёса) без Blender / Autodesk FBX SDK.
Формат: https://code.blender.org/2013/08/fbx-binary-file-format-specification/
"""

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MAGIC = b"Kaydara FBX Binary  \x00"
_ARRAY_TYPES = {"f": ("f", 4), "d": ("d", 8), "l": ("q", 8), "i": ("i", 4), "b": ("?", 1)}


@dataclass
class FbxNode:
    name: str
    properties: list
    children: list["FbxNode"] = field(default_factory=list)

    def find(self, name: str) -> "FbxNode | None":
        return next((c for c in self.children if c.name == name), None)

    def find_all(self, name: str) -> list["FbxNode"]:
        return [c for c in self.children if c.name == name]


def _read_property(data: bytes, pos: int):
    kind = chr(data[pos])
    pos += 1
    scalars = {"Y": ("h", 2), "C": ("?", 1), "I": ("i", 4), "F": ("f", 4), "D": ("d", 8), "L": ("q", 8)}
    if kind in scalars:
        fmt, size = scalars[kind]
        return struct.unpack_from("<" + fmt, data, pos)[0], pos + size
    if kind in _ARRAY_TYPES:
        length, encoding, compressed = struct.unpack_from("<III", data, pos)
        pos += 12
        raw = data[pos:pos + compressed]
        pos += compressed
        if encoding == 1:
            raw = zlib.decompress(raw)
        fmt, size = _ARRAY_TYPES[kind]
        return np.frombuffer(raw, dtype="<" + fmt, count=length).copy(), pos
    if kind in ("S", "R"):
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        raw = data[pos:pos + length]
        return (raw.decode("utf-8", "replace") if kind == "S" else raw), pos + length
    raise ValueError(f"Неизвестный тип свойства FBX: {kind!r}")


def _read_node(data: bytes, pos: int, wide: bool):
    if wide:
        end, count, _, name_len = struct.unpack_from("<QQQB", data, pos)
        pos += 25
    else:
        end, count, _, name_len = struct.unpack_from("<IIIB", data, pos)
        pos += 13
    if end == 0:
        return None, pos
    name = data[pos:pos + name_len].decode("ascii", "replace")
    pos += name_len
    properties = []
    for _ in range(count):
        value, pos = _read_property(data, pos)
        properties.append(value)
    node = FbxNode(name, properties)
    sentinel = 25 if wide else 13
    while pos < end - sentinel:
        child, pos = _read_node(data, pos, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end


def read_fbx(path: Path) -> FbxNode:
    data = Path(path).read_bytes()
    if not data.startswith(_MAGIC):
        raise ValueError(f"{path}: не бинарный FBX (ASCII FBX не поддерживается)")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    root = FbxNode("root", [version])
    pos = 27
    while pos < len(data):
        node, pos = _read_node(data, pos, wide)
        if node is None:
            break
        root.children.append(node)
    return root


def properties70(node: FbxNode | None) -> dict:
    """Properties70 узла: {имя: значения}."""
    if node is None:
        return {}
    block = node.find("Properties70")
    if block is None:
        return {}
    return {p.properties[0]: p.properties[4:] for p in block.find_all("P")}


def _euler_xyz(degrees) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(degrees, dtype=float))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                        # порядок по умолчанию в FBX: XYZ (сначала X)


@dataclass
class FbxMesh:
    name: str
    vertices: np.ndarray                       # (N, 3) в системе модели FBX, с учётом иерархии трансформаций
    local_vertices: np.ndarray                 # (N, 3) как в файле


@dataclass
class FbxScene:
    meshes: list[FbxMesh]
    global_settings: dict
    model_names: dict[int, str]


def load_meshes(path: Path) -> FbxScene:
    root = read_fbx(path)
    objects = root.find("Objects")
    geometries, models = {}, {}
    for node in objects.children:
        uid = node.properties[0]
        name = str(node.properties[1]).split("\x00")[0]
        if node.name == "Geometry" and node.find("Vertices") is not None:
            geometries[uid] = (name, node.find("Vertices").properties[0].reshape(-1, 3).astype(float))
        elif node.name == "Model":
            p = properties70(node)
            T = np.eye(4)
            R = _euler_xyz(p.get("Lcl Rotation", (0, 0, 0)))
            S = np.diag(np.asarray(p.get("Lcl Scaling", (1, 1, 1)), dtype=float))
            pre = _euler_xyz(p.get("PreRotation", (0, 0, 0)))
            T[:3, :3] = pre @ R @ S
            T[:3, 3] = np.asarray(p.get("Lcl Translation", (0, 0, 0)), dtype=float)
            models[uid] = (name, T)

    parent: dict[int, int] = {}
    geometry_model: dict[int, int] = {}
    for c in root.find("Connections").find_all("C"):
        if c.properties[0] != "OO":
            continue
        child, par = c.properties[1], c.properties[2]
        if child in geometries and par in models:
            geometry_model[child] = par
        elif child in models:
            parent[child] = par

    def world(model_id: int) -> np.ndarray:
        T = models[model_id][1]
        p = parent.get(model_id)
        return world(p) @ T if p in models else T

    meshes = []
    for gid, (gname, verts) in geometries.items():
        mid = geometry_model.get(gid)
        T = world(mid) if mid is not None else np.eye(4)
        name = models[mid][0] if mid is not None else gname
        transformed = verts @ T[:3, :3].T + T[:3, 3]
        meshes.append(FbxMesh(name, transformed, verts))
    settings = properties70(root.find("GlobalSettings"))
    return FbxScene(meshes, settings, {k: v[0] for k, v in models.items()})
