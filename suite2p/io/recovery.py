import os
import datetime
import shutil

def recover_tiff(file, tmp='/tmp'):
    print('attempting to recover corrupted tif: ' + file)
    recovery = 'suite2p_tiff_recovery_' + datetime.datetime.now().strftime("%y%m%d%H%M%S%f") + '.tif'
    recovery = os.path.join(tmp, recovery)
    
    with open(file, mode='rb') as binary:
        binary.seek(0, os.SEEK_END)
        eof = binary.tell()
        
        offset = int('08', base=16)
        binary.seek(offset)
        ifd = binary.read(8).hex()
        ifd = ''.join([a + b for a, b in zip(ifd[-2::-2], ifd[-1::-2])])
        ifd = int(ifd, base=16)
    
        count = 0
        last = 0
        while ifd < eof:
            binary.seek(ifd)
            ntags = binary.read(8).hex()
            ntags = ''.join([a + b for a, b in zip(ntags[-2::-2], ntags[-1::-2])])
            ntags = int(ntags, base=16)
    
            last = offset
            offset = ifd + 8 + ntags * 20
            binary.seek(offset)
            ifd = binary.read(8).hex()
            ifd = ''.join([a + b for a, b in zip(ifd[-2::-2], ifd[-1::-2])])
            ifd = int(ifd, base=16)
            
            if ifd == 0 or ifd > eof:
                break
                
            count += 1
    
    shutil.copyfile(file, recovery)
    with open(recovery, mode='r+b') as binary:
        binary.seek(last)
        binary.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    
    print('successfully recovered ' + str(count) + ' frames')
    print('recovery saved at ' + recovery)
    print('can now pass it on to ScanImageTiffReader')

    return recovery, count

