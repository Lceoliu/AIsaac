// 离线验证 shader_registry.hpp：哈希、红黑树查找、shader 栈出栈算术。全部用伪造内存，不需要游戏。
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../src/shader_registry.hpp"

using namespace isaac_turbo;

namespace {

struct Node {
  std::uint32_t left, parent, right;
  std::uint8_t color, isnil, pad[2];
  std::uint32_t key, value;
};
static_assert(sizeof(Node) == 0x18, "node layout");

struct Shader {
  std::uint32_t vtable, flags;
};

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  check(shaderNameHash("KAGE_ColorTextureShader") == kColorTextureShaderHash, "hash KAGE_ColorTextureShader == 0xB3D14323");
  check(shaderNameHash("kage_colortextureshader") == kColorTextureShaderHash, "hash is case-insensitive");
  check(shaderNameHash("KAGE_ColorShader") != kColorTextureShaderHash, "different names differ");

  // 三个内置 shader：键排序后为 0x4B73AA92 < 0x7020A425 < 0xB3D14323（09-19 活体读取到的注册表键）
  Shader colorShader{0x1000, 1}, colorTexture{0x1000, 1}, indexed{0x1000, 1};
  Node head{}, a{}, b{}, c{};
  head.isnil = 1;
  head.color = 1;
  auto addr = [](const void* p) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(p)); };
  b.key = 0x7020A425u; b.value = addr(&colorShader); b.parent = addr(&head); b.left = addr(&a); b.right = addr(&c);
  a.key = 0x4B73AA92u; a.value = addr(&indexed); a.parent = addr(&b); a.left = addr(&head); a.right = addr(&head);
  c.key = 0xB3D14323u; c.value = addr(&colorTexture); c.parent = addr(&b); c.left = addr(&head); c.right = addr(&head);
  head.parent = addr(&b);  // root
  head.left = addr(&a);
  head.right = addr(&c);
  std::uint32_t mapObject[2] = {addr(&head), 3};
  const std::uint8_t* mapBytes = reinterpret_cast<const std::uint8_t*>(mapObject);

  RegistryLookup r = lookupShaderInRegistry(mapBytes, kColorTextureShaderHash);
  check(r.found && r.object == addr(&colorTexture) && r.flags == 1 && r.size == 3, "lookup finds ColorTextureShader with flags=1");
  r = lookupShaderInRegistry(mapBytes, 0x4B73AA92u);
  check(r.found && r.object == addr(&indexed), "lookup finds smallest key");
  r = lookupShaderInRegistry(mapBytes, 0x12345678u);
  check(!r.found && r.size == 3, "missing key is not found, size still reported");
  colorTexture.flags = 0;
  r = lookupShaderInRegistry(mapBytes, kColorTextureShaderHash);
  check(r.found && r.flags == 0, "in-place invalidated object reports flags=0 (push would fail)");
  std::uint32_t emptyMap[2] = {0, 0};
  r = lookupShaderInRegistry(reinterpret_cast<const std::uint8_t*>(emptyMap), kColorTextureShaderHash);
  check(!r.found && r.size == 0, "null head handled");

  // 出栈算术：崩溃转储里 offset=0,size=0,mapsize=8，块 7 为空 -> 读地址 0xC（EAX=0xC, ESI=0xFFFFFFFF）
  std::uint32_t deque[5] = {0, 0xDEAD0000u, 8, 0, 0};
  std::uint32_t blocks[8] = {};
  ShaderStackState s = readShaderStack(reinterpret_cast<const std::uint8_t*>(deque));
  check(s.mapsize == 8 && s.size == 0 && s.offset == 0, "deque fields");
  check(popSlotAddress(s, blocks) == 0xCu, "empty-stack pop reads NULL+0xC exactly as in the dumps");
  blocks[0] = 0x5000;
  deque[4] = 1;
  s = readShaderStack(reinterpret_cast<const std::uint8_t*>(deque));
  check(popSlotAddress(s, blocks) == 0x5000u, "depth 1 pops slot 0 of block 0");

  if (failures == 0) std::printf("PASS registry_test\n");
  return failures == 0 ? 0 : 1;
}
