import ctypes
import hashlib
import json


def release_unused_cpu_memory():
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return bool(trim(0))


class CachedVolumeDecoder:

    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, latents, geo_decoder, **kwargs):
        attention = geo_decoder.cross_attn_decoder.attn
        previous_setting = attention.kv_cache
        attention.data = None
        attention.kv_cache = True
        try:
            return self.decoder(latents, geo_decoder, **kwargs)
        finally:

            attention.data = None
            attention.kv_cache = previous_setting


def reconstruction_key(image, **settings):
    metadata = {"version": 1, "mode": image.mode, "size": image.size, "settings": settings}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()
