# Native Mojo planner for the flat C ABI in runtime_split.h. This does not
# execute expert kernels: the C dispatcher owns state, concurrency, and joins.
from std.memory import Pointer
from std.runtime import initialize_runtime

comptime ReadU32 = Optional[Pointer[UInt32, ImmUntrackedOrigin]]
comptime ReadF64 = Optional[Pointer[Float64, ImmUntrackedOrigin]]
comptime WriteU32 = Optional[Pointer[UInt32, MutUntrackedOrigin]]
comptime WriteF64 = Optional[Pointer[Float64, MutUntrackedOrigin]]

comptime FRAC = UInt64(0x000fffffffffffff)
comptime MAG = UInt64(0x7fffffffffffffff)
comptime EXP = UInt64(0x7ff0000000000000)
comptime HIDDEN = UInt64(0x0010000000000000)

def shift_jam(value: UInt64, shift: Int) -> UInt64:
    if shift == 0:
        return value
    if shift >= 64:
        return UInt64(value != 0)
    return (value >> UInt64(shift)) | UInt64((value & ((UInt64(1) << UInt64(shift)) - 1)) != 0)

def positive_add(left: UInt64, right: UInt64) -> UInt64:
    # Integer binary64 arithmetic preserves subnormals and ties under FTZ.
    # The infinity bit pattern is a sticky overflow sentinel, never a result.
    if (left & EXP) == EXP or (right & EXP) == EXP:
        return EXP
    if left == 0:
        return right
    if right == 0:
        return left
    var a = max(left, right)
    var b = min(left, right)
    var ea = Int(a >> 52)
    var eb = Int(b >> 52)
    var ma = a & FRAC
    var mb = b & FRAC
    if ea > 0:
        ma |= HIDDEN
    else:
        ea = 1
    if eb > 0:
        mb |= HIDDEN
    else:
        eb = 1
    var total = (ma << 3) + shift_jam(mb << 3, ea - eb)
    if total >= (HIDDEN << 4):
        total = shift_jam(total, 1)
        ea += 1
    var significand = total >> 3
    var tail = total & 7
    if tail > 4 or (tail == 4 and (significand & 1) != 0):
        significand += 1
    if significand >= (HIDDEN << 1):
        significand >>= 1
        ea += 1
    if ea >= 2047:
        return EXP
    if ea == 1 and significand < HIDDEN:
        return significand
    return (UInt64(ea) << 52) | (significand & FRAC)

def high_bit(value: UInt64) -> Int:
    var remaining = value
    var bit = 0
    while remaining > 1:
        remaining >>= 1
        bit += 1
    return bit

def multiply_count(value: UInt64, count: Int) -> UInt64:
    if value == 0 or count == 0:
        return 0
    var exponent = Int(value >> 52)
    var significand = value & FRAC
    if exponent > 0:
        significand |= HIDDEN
    else:
        exponent = 1
    # Count <= 1024: the exact significand product fits uint64.
    var product = significand * UInt64(count)
    var high = high_bit(product)
    if high > 52:
        var shift = high - 52
        var tail = product & ((UInt64(1) << UInt64(shift)) - 1)
        var half = UInt64(1) << UInt64(shift - 1)
        significand = product >> UInt64(shift)
        if tail > half or (tail == half and (significand & 1) != 0):
            significand += 1
        exponent += shift
    else:
        significand = product
    if significand >= (HIDDEN << 1):
        significand >>= 1
        exponent += 1
    if exponent >= 2047:
        return EXP
    if exponent == 1 and significand < HIDDEN:
        return significand
    return (UInt64(exponent) << 52) | (significand & FRAC)

@export("coli_mojo_split_abi")
def split_abi() abi("C") -> UInt32:
    return 1

@export("coli_mojo_split_init")
def split_init() abi("C") -> Int32:
    initialize_runtime()
    return 0

@export("coli_mojo_split_plan")
def split_plan(
    controls_in: ReadU32, costs_in: ReadF64,
    counts_out: WriteU32, times_out: WriteF64,
) abi("C") -> Int32:
    if not controls_in or not costs_in or not counts_out or not times_out:
        return 1
    var controls = controls_in.value()
    var costs = costs_in.value()
    var counts = counts_out.value()
    var times = times_out.value()
    var counts_addr = UInt(Int(counts))
    var times_addr = UInt(Int(times))
    if counts_addr > ~UInt(0) - 8 or times_addr > ~UInt(0) - 32:
        return 1
    if counts_addr <= times_addr:
        if times_addr - counts_addr < 8:
            return 1
    else:
        if counts_addr - times_addr < 32:
            return 1
    var abi = controls[]
    var misses = Int(controls[unsafe_offset=1])
    var maximum = Int(controls[unsafe_offset=2])
    var overlap = controls[unsafe_offset=3]
    if abi != 1 or misses > 1024 or maximum > misses or overlap > 1:
        return 1
    var cost = Array[UInt64, 9](uninitialized=True)
    for i in range(9):
        var value = costs[unsafe_offset=i]
        var bits = Pointer(to=value).unsafe_bitcast[UInt64]()[]
        if (bits & EXP) == EXP or ((bits >> 63) != 0 and (bits & MAG) != 0):
            return 1
        cost[i] = bits & MAG
    var best_q = -1
    var best_host = UInt64(0)
    var best_device = UInt64(0)
    var best_serial = UInt64(0)
    var best_total = UInt64(0)
    for q in range(maximum + 1):
        var h = misses - q
        if (h > 0 and cost[0] == 0) or (q > 0 and cost[2] == 0):
            continue
        var host = positive_add(multiply_count(cost[0], h), cost[3])
        var device = positive_add(multiply_count(cost[2], q), cost[4])
        var serial = cost[8]
        if h > 0 or cost[3] > 0:
            host = positive_add(host, cost[5])
        if q > 0 or cost[4] > 0:
            device = positive_add(device, cost[6])
        var transfer = multiply_count(cost[1], q)
        if overlap == 1:
            device = positive_add(device, transfer)
        else:
            serial = positive_add(serial, transfer)
        var total = positive_add(positive_add(serial, max(host, device)), cost[7])
        if (total & EXP) == EXP:
            continue
        if best_q < 0 or total < best_total:
            best_q = q
            best_host = host
            best_device = device
            best_serial = serial
            best_total = total
    if best_q < 0:
        return 2
    # Publish once, after every validation and candidate decision has succeeded.
    counts[] = UInt32(best_q)
    counts[unsafe_offset=1] = UInt32(misses - best_q)
    times.unsafe_bitcast[UInt64]()[] = best_host
    times.unsafe_bitcast[UInt64]()[unsafe_offset=1] = best_device
    times.unsafe_bitcast[UInt64]()[unsafe_offset=2] = best_serial
    times.unsafe_bitcast[UInt64]()[unsafe_offset=3] = best_total
    return 0
