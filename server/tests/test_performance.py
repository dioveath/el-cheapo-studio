from types import SimpleNamespace
import unittest

from PIL import Image

from studio.pipeline.performance import CachedVolumeDecoder, reconstruction_key


class PerformanceTests(unittest.TestCase):
    def test_reconstruction_key_uses_pixels_and_all_generation_settings(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        settings = dict(model="shape", revision="v1", steps=30, resolution=256,
                        seed=42, remove_background=True, decode_chunk_size=8000)
        reference = reconstruction_key(image, **settings)
        self.assertEqual(reference, reconstruction_key(image.copy(), **dict(reversed(list(settings.items())))))
        changes = dict(model="other", revision="v2", steps=20, resolution=128,
                       seed=43, remove_background=False, decode_chunk_size=4000)
        for key, value in changes.items():
            self.assertNotEqual(reference, reconstruction_key(image, **(settings | {key: value})))
        image.putpixel((1, 1), (0, 0, 0, 255))
        self.assertNotEqual(reference, reconstruction_key(image, **settings))

    def test_decoder_cache_is_scoped_to_one_image_and_cleared_on_failure(self):
        attention = SimpleNamespace(kv_cache=False, data="stale")
        geo = SimpleNamespace(cross_attn_decoder=SimpleNamespace(attn=attention))

        def decode(latents, geo_decoder, fail=False):
            self.assertTrue(attention.kv_cache)
            self.assertIsNone(attention.data)
            attention.data = latents * 2
            if fail:
                raise RuntimeError("simulated decode failure")
            return attention.data

        cached = CachedVolumeDecoder(decode)
        self.assertEqual(cached(3, geo), 6)
        self.assertIsNone(attention.data)
        self.assertFalse(attention.kv_cache)
        self.assertEqual(cached(5, geo), 10)
        with self.assertRaises(RuntimeError):
            cached(7, geo, fail=True)
        self.assertIsNone(attention.data)
        self.assertFalse(attention.kv_cache)
        self.assertEqual(cached(11, geo), 22)


if __name__ == "__main__":
    unittest.main()
