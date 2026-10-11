import math
import random
import struct
import unittest

import heads
import precision


class PrecisionTests(unittest.TestCase):
    def test_randomized_error_storage_and_canonicality(self):
        rng = random.Random(20261010)
        for bits in (4, 8, 16):
            for block in (1, 3, 16, 64):
                for n in (0, 1, 2, 3, 17, 129):
                    values = [precision.f32(rng.uniform(-10, 10)) for _ in range(n)]
                    blob = precision.encode(values, bits, block)
                    info, decoded = precision.decode(blob)
                    self.assertEqual(info["bits"], bits)
                    self.assertEqual(info["count"], n)
                    self.assertEqual(precision.encode(decoded, bits, block), blob)
                    for start in range(0, n, block):
                        maximum = max(map(abs, values[start:start + block]))
                        bound = maximum / (2 * ((1 << (bits - 1)) - 1)) + maximum * 2**-23
                        for v, w in zip(values[start:start + block], decoded[start:start + block]):
                            self.assertLessEqual(abs(v - w), bound)

    def test_extrema_and_invalid_input(self):
        for bits in (4, 8, 16):
            values = [heads.FLOAT32_MAX, -heads.FLOAT32_MAX, 0., -0.]
            _, decoded = precision.decode(precision.encode(values, bits, 2))
            self.assertEqual(decoded, values)
            smallest = struct.unpack("<f", struct.pack("<I", 1))[0]
            self.assertEqual(precision.decode(precision.encode([smallest, -smallest], bits, 2))[1],
                             [smallest, -smallest])
            for v in (math.nan, math.inf, -math.inf, 10**400, True):
                with self.assertRaises(ValueError):
                    precision.encode([v], bits)

    def test_malformed_and_cross_language_refusal_contract(self):
        for bits in (4, 8, 16):
            valid = precision.encode([1., 0., -1.], bits, 3)
            invalid = [valid[:n] for n in range(len(valid))] + [valid + b"x"]
            for offset, data in ((0, b"nope"), (4, struct.pack("<I", 2)),
                                 (8, struct.pack("<I", 3)), (16, b"\0" * 4),
                                 (24, struct.pack("<f", math.nan)), (24, struct.pack("<f", -1.))):
                bad = bytearray(valid); bad[offset:offset + len(data)] = data; invalid.append(bytes(bad))
            zero = bytearray(precision.encode([0., 0.], bits, 2))
            zero[24:28] = struct.pack("<I", 0x80000000)
            invalid.append(bytes(zero))
            no_extreme = bytearray(valid); no_extreme[28:] = bytes(len(valid) - 28)
            invalid.append(bytes(no_extreme))
            if bits == 4:
                padding = bytearray(valid); padding[-1] |= 0x10; invalid.append(bytes(padding))
            for blob in invalid:
                with self.assertRaises(ValueError):
                    precision.decode(blob)

    def test_int16_is_integer_and_int4_packing(self):
        self.assertEqual(precision.encode([1., -1., 0.], 4, 3)[28:], bytes((0x97, 0)))
        self.assertEqual(precision.encode([1., -1.], 16, 2)[28:], bytes((0xff, 0x7f, 1, 0x80)))


if __name__ == "__main__":
    unittest.main()
