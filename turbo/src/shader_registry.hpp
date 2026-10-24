// J460 KAGE shader 注册表 / shader 栈的只读解析，纯逻辑、无 Windows 依赖，可在 tests/registry_test.cpp 里
// 用伪造的内存布局验证。布局来自反编译（rl/docs/NATIVE_CRASH_ANALYSIS.md §2.1）：
//   注册表 std::map<u32 hash, Shader*>：对象 {+0 head, +4 size}；节点 {+0 left, +4 parent, +8 right,
//   +0xC color, +0xD isnil, +0x10 key, +0x14 value}；head->parent 是根。
//   Shader 对象 +4 是 flags，bit0 = Shader::Create 成功后置位；入栈函数 FUN_00a140c0 只接受 bit0 为 1 的对象。
//   shader 栈 std::deque<Shader*>：{+4 map, +8 mapsize, +0xC offset, +0x10 size}，每块 4 个指针。
//   键 = 对小写名字的 djb2：h = 5381; h = h*33 + c。
#pragma once

#include <cstddef>
#include <cstdint>

namespace isaac_turbo {

inline std::uint32_t shaderNameHash(const char* name) {
  std::uint32_t h = 5381u;
  for (; name && *name; ++name) {
    unsigned char c = static_cast<unsigned char>(*name);
    if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + 32);
    h = h * 33u + c;
  }
  return h;
}

constexpr std::uint32_t kColorTextureShaderHash = 0xB3D14323u;  // shaderNameHash("KAGE_ColorTextureShader")

struct RegistryLookup {
  bool found = false;
  std::uint32_t size = 0;    // 注册表元素数
  std::uint32_t node = 0;    // 命中的节点地址
  std::uint32_t object = 0;  // 节点值（Shader*）
  std::uint32_t flags = 0;   // 对象 +4
  std::uint32_t steps = 0;   // 遍历深度
};

// 在给定进程内存里按哈希查找。mapObject 指向注册表对象（RVA 0x8379BC 处）。
// 所有指针都按本进程可直接解引用处理；调用方负责只在能安全读取时调用。
inline RegistryLookup lookupShaderInRegistry(const std::uint8_t* mapObject, std::uint32_t hash) {
  RegistryLookup r;
  const std::uint32_t head = *reinterpret_cast<const std::uint32_t*>(mapObject);
  r.size = *reinterpret_cast<const std::uint32_t*>(mapObject + 4);
  if (head == 0) return r;
  std::uint32_t node = *reinterpret_cast<const std::uint32_t*>(head + 4);  // root
  while (node != 0 && node != head && r.steps < 64) {
    ++r.steps;
    const std::uint8_t* n = reinterpret_cast<const std::uint8_t*>(node);
    if (n[0xD] != 0) break;  // isnil
    const std::uint32_t key = *reinterpret_cast<const std::uint32_t*>(n + 0x10);
    if (hash < key) {
      node = *reinterpret_cast<const std::uint32_t*>(n + 0);
    } else if (key < hash) {
      node = *reinterpret_cast<const std::uint32_t*>(n + 8);
    } else {
      r.found = true;
      r.node = node;
      r.object = *reinterpret_cast<const std::uint32_t*>(n + 0x14);
      if (r.object) r.flags = *reinterpret_cast<const std::uint32_t*>(r.object + 4);
      return r;
    }
  }
  return r;
}

struct ShaderStackState {
  std::uint32_t map = 0, mapsize = 0, offset = 0, size = 0;
};

inline ShaderStackState readShaderStack(const std::uint8_t* dequeObject) {
  ShaderStackState s;
  s.map = *reinterpret_cast<const std::uint32_t*>(dequeObject + 4);
  s.mapsize = *reinterpret_cast<const std::uint32_t*>(dequeObject + 8);
  s.offset = *reinterpret_cast<const std::uint32_t*>(dequeObject + 0xC);
  s.size = *reinterpret_cast<const std::uint32_t*>(dequeObject + 0x10);
  return s;
}

// 字体函数出栈时读取的槽地址（FUN_00684fc0 的算术）：index = offset + size - 1；
// 槽 = map[(index >> 2) & (mapsize - 1)] + (index & 3) * 4。size == 0 且块为空时得到 0xC，即崩溃读地址。
inline std::uint32_t popSlotAddress(const ShaderStackState& s, const std::uint32_t* blocks) {
  const std::uint32_t index = s.offset + s.size - 1u;
  const std::uint32_t blockIndex = (index >> 2) & (s.mapsize - 1u);
  const std::uint32_t block = blocks ? blocks[blockIndex] : 0u;
  return block + (index & 3u) * 4u;
}

}  // namespace isaac_turbo
