# Copyright (c) 2016-2017,2019-2021 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""This script is the full system example script from the ARM
Research Starter Kit on System Modeling. More information can be found
at: http://www.arm.com/ResearchEnablement/SystemModeling
"""

import os
import m5
from m5.util import addToPath
from m5.objects import *
from m5.options import *
import argparse

m5.util.addToPath('../..')

from common import SysPaths
from common import MemConfig
from common import ObjectList
from common.cores.arm import HPI
from m5.util.fdthelper import *
import workloads
import m5
from common.Caches import *
from common import ObjectList

have_kvm = "ArmV8KvmCPU" in ObjectList.cpu_list.get_names()
have_fastmodel = "FastModelCortexA76" in ObjectList.cpu_list.get_names()

class L1I(L1_ICache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 8
    size = '48kB'
    assoc = 3


class L1D(L1_DCache):
    tag_latency = 2
    data_latency = 2
    response_latency = 1
    mshrs = 16
    tgts_per_mshr = 16
    size = '32kB'
    assoc = 2
    write_buffers = 16


class WalkCache(PageTableWalkerCache):
    tag_latency = 4
    data_latency = 4
    response_latency = 4
    mshrs = 6
    tgts_per_mshr = 8
    size = '1kB'
    assoc = 8
    write_buffers = 16


class L2(L2Cache):
    tag_latency = 12
    data_latency = 12
    response_latency = 5
    mshrs = 32
    tgts_per_mshr = 8
    size = '1MB'
    assoc = 16
    write_buffers = 8
    clusivity='mostly_excl'


class L3(Cache):
    size = '16MB'
    assoc = 16
    tag_latency = 20
    data_latency = 20
    response_latency = 20
    mshrs = 20
    tgts_per_mshr = 12
    clusivity='mostly_excl'


class MemBus(SystemXBar):
    badaddr_responder = BadAddr(warn_access="warn")
    default = Self.badaddr_responder.pio

def simpleSystem(BaseSystem, caches, mem_size, platform=None, **kwargs):
    """
    Create a simple system example.  The base class in configurable so
    that it is possible (e.g) to link the platform (hardware configuration)
    with a baremetal ArmSystem or with a LinuxArmSystem.
    """
    class SimpleSystem(BaseSystem):
        cache_line_size = 64
        
        def __init__(self, caches, mem_size, platform=None, **kwargs):
            super(SimpleSystem, self).__init__(**kwargs)

            self.voltage_domain = VoltageDomain(voltage="1.0V")
            self.clk_domain = SrcClockDomain(
                clock="1GHz",
                voltage_domain=Parent.voltage_domain)

            if platform is None:
                self.realview = VExpress_GEM5_V1()
            else:
                self.realview = platform

            if hasattr(self.realview.gic, 'cpu_addr'):
                self.gic_cpu_addr = self.realview.gic.cpu_addr

            self.membus = MemBus()

            self.terminal = Terminal()
            self.vncserver = VncServer()
            self._optee=None
            self.iobus = IOXBar()
            # CPUs->PIO
            self.iobridge = Bridge(delay='50ns')
            # Device DMA -> MEM
            mem_range = self.realview._mem_regions[0]
            assert int(mem_range.size()) >= int(Addr(mem_size))
            self.mem_ranges = [
                AddrRange(start=mem_range.start, size=mem_size) ]

            self._caches = caches
            if self._caches:
                self.iocache = IOCache(addr_ranges=[self.mem_ranges[0]])
            else:
                self.dmabridge = Bridge(delay='50ns',
                                        ranges=[self.mem_ranges[0]])

            self._clusters = []
            self._num_cpus = 0
            

        def attach_pci(self, dev):
            self.realview.attachPciDevice(dev, self.iobus)

        def connect(self):
            self.iobridge.mem_side_port = self.iobus.cpu_side_ports
            self.iobridge.cpu_side_port = self.membus.mem_side_ports
    
            if self._caches:
                self.iocache.mem_side = self.membus.cpu_side_ports
                self.iocache.cpu_side = self.iobus.mem_side_ports
            else:
                self.dmabridge.mem_side_port = self.membus.cpu_side_ports
                self.dmabridge.cpu_side_port = self.iobus.mem_side_ports
    
            if hasattr(self.realview.gic, 'cpu_addr'):
                self.gic_cpu_addr = self.realview.gic.cpu_addr
            self.realview.attachOnChipIO(self.membus, self.iobridge)
            self.realview.attachIO(self.iobus)
            self.system_port = self.membus.cpu_side_ports

        def numCpuClusters(self):
            return len(self._clusters)

        def addCpuCluster(self, cpu_cluster, num_cpus):
            assert cpu_cluster not in self._clusters
            assert num_cpus > 0
            self._clusters.append(cpu_cluster)
            self._num_cpus += num_cpus

        def numCpus(self):
            return self._num_cpus

        def addCaches(self, need_caches, last_cache_level):
            if not need_caches:
                # connect each cluster to the memory hierarchy
                for cluster in self._clusters:
                    cluster.connectMemSide(self.membus)
                return

            cluster_mem_bus = self.membus
            assert last_cache_level >= 1 and last_cache_level <= 3
            for cluster in self._clusters:
                cluster.addL1()
            if last_cache_level > 1:
                for cluster in self._clusters:
                    cluster.addL2(cluster.clk_domain)
            if last_cache_level > 2:
                max_clock_cluster = max(self._clusters,
                                        key=lambda c: c.clk_domain.clock[0])
                self.l3 = L3(clk_domain=max_clock_cluster.clk_domain)
                self.toL3Bus = L2XBar(width=64)
                self.toL3Bus.mem_side_ports = self.l3.cpu_side
                self.l3.mem_side = self.membus.cpu_side_ports
                cluster_mem_bus = self.toL3Bus

            # connect each cluster to the memory hierarchy
            for cluster in self._clusters:
                cluster.connectMemSide(cluster_mem_bus)

    return SimpleSystem(caches, mem_size, platform, **kwargs)


class CpuCluster(SubSystem):
    def __init__(self, system,  num_cpus, cpu_clock, cpu_voltage,
                 cpu_type, l1i_type, l1d_type, wcache_type, l2_type):
        super(CpuCluster, self).__init__()
        self._cpu_type = cpu_type
        self._l1i_type = l1i_type
        self._l1d_type = l1d_type
        self._wcache_type = wcache_type
        self._l2_type = l2_type

        assert num_cpus > 0
        
        self.voltage_domain = VoltageDomain(voltage=cpu_voltage)
        self.clk_domain = SrcClockDomain(clock=cpu_clock,
                                         voltage_domain=self.voltage_domain)

        self.cpus = [ self._cpu_type(cpu_id=system.numCpus() + idx,
                                     clk_domain=self.clk_domain)
                      for idx in range(num_cpus) ]

        for cpu in self.cpus:
            cpu.createThreads()
            cpu.createInterruptController()
            cpu.socket_id = system.numCpuClusters()
        system.addCpuCluster(self, num_cpus)

    def requireCaches(self):
        return self._cpu_type.require_caches()

    def memoryMode(self):
        return self._cpu_type.memory_mode()

    def addL1(self):
        for cpu in self.cpus:
            l1i = None if self._l1i_type is None else self._l1i_type()
            l1d = None if self._l1d_type is None else self._l1d_type()
            iwc = None if self._wcache_type is None else self._wcache_type()
            dwc = None if self._wcache_type is None else self._wcache_type()
            cpu.addPrivateSplitL1Caches(l1i, l1d, iwc, dwc)

    def addL2(self, clk_domain):
        if self._l2_type is None:
            return
        self.toL2Bus = L2XBar(width=64, clk_domain=clk_domain)
        self.l2 = self._l2_type()
        for cpu in self.cpus:
            cpu.connectAllPorts(self.toL2Bus)
        self.toL2Bus.mem_side_ports = self.l2.cpu_side

    def addPMUs(self, ints, events=[]):
        """
        Instantiates 1 ArmPMU per PE. The method is accepting a list of
        interrupt numbers (ints) used by the PMU and a list of events to
        register in it.

        :param ints: List of interrupt numbers. The code will iterate over
            the cpu list in order and will assign to every cpu in the cluster
            a PMU with the matching interrupt.
        :type ints: List[int]
        :param events: Additional events to be measured by the PMUs
        :type events: List[Union[ProbeEvent, SoftwareIncrement]]
        """
        assert len(ints) == len(self.cpus)
        for cpu, pint in zip(self.cpus, ints):
            int_cls = ArmPPI if pint < 32 else ArmSPI
            for isa in cpu.isa:
                isa.pmu = ArmPMU(interrupt=int_cls(num=pint))
                isa.pmu.addArchEvents(cpu=cpu,
                                      itb=cpu.mmu.itb, dtb=cpu.mmu.dtb,
                                      icache=getattr(cpu, 'icache', None),
                                      dcache=getattr(cpu, 'dcache', None),
                                      l2cache=getattr(self, 'l2', None))
                for ev in events:
                    isa.pmu.addEvent(ev)

    def connectMemSide(self, bus):
        try:
            self.l2.mem_side = bus.cpu_side_ports
        except AttributeError:
            for cpu in self.cpus:
                cpu.connectAllPorts(bus)
class OPTEE_firmware(SubSystem):
    def __init__(self, *args, **kwargs):
        super(OPTEE_firmware, self).__init__(*args,**kwargs)
        pass

    def generateDeviceTree(self, state):
        node = FdtNode("firmware")
        node_subnode = FdtNode("optee")
        node_subnode.appendCompatible("linaro,optee-tz")
        node_subnode.append(FdtPropertyStrings("method",["smc"]))
        node.append(node_subnode)

        yield node


# Pre-defined CPU configurations. Each tuple must be ordered as : (cpu_class,
# l1_icache_class, l1_dcache_class, walk_cache_class, l2_Cache_class). Any of
# the cache class may be 'None' if the particular cache is not present.
cpu_types = {

    "atomic" : ( AtomicSimpleCPU, None, None, None, None),
    "minor" : (MinorCPU,
               L1I, L1D,
               WalkCache,
               L2),
    "hpi" : ( HPI.HPI,
              HPI.HPI_ICache, HPI.HPI_DCache,
              HPI.HPI_WalkCache,
              HPI.HPI_L2)
}

def create_cow_image(name):
    """Helper function to create a Copy-on-Write disk image"""
    image = CowDiskImage()
    image.child.image_file = name
    return image;


def create(args):
    ''' Create and configure the system object. '''

    if args.readfile and not os.path.isfile(args.readfile):
        print("Error: Bootscript %s does not exist" % args.readfile)
        sys.exit(1)

    object_file = args.kernel if args.kernel else ""

    cpu_class = cpu_types[args.cpu][0]
    mem_mode = cpu_class.memory_mode()
    # Only simulate caches when using a timing CPU (e.g., the HPI model)
    want_caches = True if mem_mode == "timing" else False

    platform = ObjectList.platform_list.get(args.machine_type)

    system = simpleSystem(ArmSystem,
                                  want_caches,
                                  args.mem_size,
                                  platform=platform(),
                                  mem_mode=mem_mode,
                                  readfile=args.readfile)

    MemConfig.config_mem(args, system)
    
    if args.semi_enable:
        system.semihosting = ArmSemihosting(
            stdin=args.semi_stdin,
            stdout=args.semi_stdout,
            stderr=args.semi_stderr,
            files_root_dir=args.semi_path,
            cmd_line = " ".join([ object_file ] + args.args)
        )
    
    if args.disk_image:
        # Create a VirtIO block device for the system's boot
        # disk. Attach the disk image using gem5's Copy-on-Write
        # functionality to avoid writing changes to the stored copy of
        # the disk image.
        system.realview.vio[0].vio = VirtIOBlock(
            image=create_cow_image(args.disk_image))
    if args.secondary_disk:
        system.realview.vio[1].vio = VirtIOBlock(
            image=create_cow_image(args.secondary_disk))
    # Wire up the system's memory system
    system.connect()

    # Add CPU clusters to the system
    system.cpu_cluster = [
        CpuCluster(system,
                           args.num_cores,
                           args.cpu_freq, "1.0V",
                           *cpu_types[args.cpu]),
    ]
    #system.cpu_cluster[0].addPMUs([i for i in range(92,92+args.num_cores)])
    system.optee=OPTEE_firmware()
    # Create a cache hierarchy for the cluster. We are assuming that
    # clusters have core-private L1 caches and an L2 that's shared
    # within the cluster.
    for cluster in system.cpu_cluster:
        system.addCaches(want_caches, last_cache_level=2)
    
    # Setup gem5's minimal Linux boot loader.
    system.auto_reset_addr = True

    # Using GICv3
    system.realview.gic.gicv4 = False

    system.highest_el_is_64 = True
    system.have_virtualization = True
    system.have_security = True
    
    workload_class = workloads.workload_list.get(args.workload)
    system.workload = workload_class(
        object_file, system)
    
    return system

def run(args):
    cptdir = m5.options.outdir
    if args.checkpoint:
        print("Checkpoint directory: %s" % cptdir)

    while True:
        event = m5.simulate()
        exit_msg = event.getCause()
        if exit_msg == "checkpoint":
            print("Dropping checkpoint at tick %d" % m5.curTick())
            cpt_dir = os.path.join(m5.options.outdir, "cpt.%d" % m5.curTick())
            m5.checkpoint(os.path.join(cpt_dir))
            print("Checkpoint done.")
        else:
            print(exit_msg, " @ ", m5.curTick())
            break

    sys.exit(event.getCode())


def main():
    parser = argparse.ArgumentParser(epilog=__doc__)

    parser.add_argument("--kernel", type=str,
                        default=None,
                        help="Binary to run")
    parser.add_argument("--workload", type=str,
                        default="ArmBaremetal",
                        choices=workloads.workload_list.get_names(),
                        help="Workload type")
    parser.add_argument("--disk-image", type=str,
                        default=None,
                        help="Disk to instantiate")
    parser.add_argument("--secondary-disk", type=str,
                        default=None,
                        help="secondary Disk to instantiate")
                        
    parser.add_argument("--readfile", type=str, default="",
                        help = "File to return with the m5 readfile command")
    parser.add_argument("--cpu", type=str, choices=list(cpu_types.keys()),
                        default="atomic",
                        help="CPU model to use")
    parser.add_argument("--cpu-freq", type=str, default="4GHz")
    parser.add_argument("--num-cores", type=int, default=1,
                        help="Number of CPU cores")
    parser.add_argument("--machine-type", type=str,
                        choices=ObjectList.platform_list.get_names(),
                        default="VExpress_GEM5_V2",
                        help="Hardware platform class")
    parser.add_argument("--mem-type", default="DDR3_1600_8x8",
                        choices=ObjectList.mem_list.get_names(),
                        help = "type of memory to use")
    parser.add_argument("--mem-channels", type=int, default=1,
                        help = "number of memory channels")
    parser.add_argument("--mem-ranks", type=int, default=None,
                        help = "number of memory ranks per channel")
    parser.add_argument("--mem-size", action="store", type=str,
                        default="2GB",
                        help="Specify the physical memory size")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--dtb-gen", action="store_true",
                        help="Doesn't run simulation, it generates a DTB only")
    parser.add_argument("--semi-enable", action="store_true",
                        help="Enable semihosting support")
    parser.add_argument("--semi-stdin", type=str, default="stdin",
                        help="Standard input for semihosting " \
                        "(default: gem5's stdin)")
    parser.add_argument("--semi-stdout", type=str, default="stdout",
                        help="Standard output for semihosting " \
                        "(default: gem5's stdout)")
    parser.add_argument("--semi-stderr", type=str, default="stderr",
                        help="Standard error for semihosting " \
                        "(default: gem5's stderr)")
    parser.add_argument('--semi-path', type=str,
                        default="",
                        help=('Search path for files to be loaded through '
                              'Arm Semihosting'))
    parser.add_argument("args", default=[], nargs="*",
                        help="Semihosting arguments to pass to benchmark")
    parser.add_argument("-P", "--param", action="append", default=[],
        help="Set a SimObject parameter relative to the root node. "
             "An extended Python multi range slicing syntax can be used "
             "for arrays. For example: "
             "'system.cpu[0,1,3:8:2].max_insts_all_threads = 42' "
             "sets max_insts_all_threads for cpus 0, 1, 3, 5 and 7 "
             "Direct parameters of the root object are not accessible, "
             "only parameters of its children.")

    args = parser.parse_args()

    root = Root(full_system=True)
    root.system = create(args)

    root.apply_config(args.param)

    if args.restore is not None:
        m5.instantiate(args.restore)
    else:
        m5.instantiate()

    if args.dtb_gen:
        # No run, autogenerate DTB and exit
        root.system.generateDtb(os.path.join(m5.options.outdir, 'system.dtb'))
    else:
        run(args)

if __name__ == "__m5_main__":
    main()
