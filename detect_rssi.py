import objc
objc.loadBundle('CoreWLAN',
                bundle_path='/System/Library/Frameworks/CoreWLAN.framework',
                module_globals=globals())
cur_interface = CWInterface.interfaceWithName_('en0')

def detect_rssi():
    networks = cur_interface.scanForNetworksWithName_error_(None, None)
    ns = networks[0]
    return {str(n.ssid()): float(n.rssi()) for n in ns.allObjects()}


if __name__ == '__main__':
    print(detect_rssi().keys())

