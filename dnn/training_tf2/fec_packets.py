import numpy as np



def write_fec_packets(filename, packets, rates=None):
    """ writes packets in binary format """
    
    assert np.dtype(np.float32).itemsize == 4
    assert np.dtype(np.int16).itemsize == 2
    
    # derive some sizes 
    num_packets             = len(packets)
    subframes_per_packet    = packets[0].shape[-2]
    num_features            = packets[0].shape[-1]
    
    # size of float is 4
    subframe_size           = num_features * 4
    packet_size             = subframe_size * subframes_per_packet + 2 # two bytes for rate
    
    version = 1
    # header size (version, header_size, num_packets, packet_size, subframe_size, subrames_per_packet, num_features)
    header_size = 14
    
    with open(filename, 'wb') as f:
        
        # header
        f.write(np.int16(version).tobytes())
        f.write(np.int16(header_size).tobytes())
        f.write(np.int16(num_packets).tobytes())
        f.write(np.int16(packet_size).tobytes())
        f.write(np.int16(subframe_size).tobytes())
        f.write(np.int16(subframes_per_packet).tobytes())
        f.write(np.int16(num_features).tobytes())
        
        # packets
        for i, packet in enumerate(packets):
            if type(rates) == type(None):
                rate = 0
            else:
                rate = rates[i]
            
            f.write(np.int16(rate).tobytes())
            
            features = np.flip(packet, axis=-2)
            f.write(features.astype(np.float32).tobytes())
            
        
def read_fec_packets(filename):
    """ reads packets from binary format """
    
    assert np.dtype(np.float32).itemsize == 4
    assert np.dtype(np.int16).itemsize == 2
    
    with open(filename, 'rb') as f:
        
        # header
        version                 = np.frombuffer(f.read(2), dtype=np.int16).item()
        header_size             = np.frombuffer(f.read(2), dtype=np.int16).item()
        num_packets             = np.frombuffer(f.read(2), dtype=np.int16).item()
        packet_size             = np.frombuffer(f.read(2), dtype=np.int16).item()
        subframe_size           = np.frombuffer(f.read(2), dtype=np.int16).item()
        subframes_per_packet    = np.frombuffer(f.read(2), dtype=np.int16).item()
        num_features            = np.frombuffer(f.read(2), dtype=np.int16).item()
        
        dummy_features          = np.zeros((1, subframes_per_packet, num_features), dtype=np.float32)
        
        # packets
        rates = []
        packets = []
        for i in range(num_packets):
                     
            rate = np.frombuffer(f.read(2), dtype=np.int16).item
            rates.append(rate)
            
            features = np.reshape(np.frombuffer(f.read(subframe_size * subframes_per_packet), dtype=np.float32), dummy_features.shape)
            packet = np.flip(features, axis=-2)
            packets.append(packet)
            
    return packets