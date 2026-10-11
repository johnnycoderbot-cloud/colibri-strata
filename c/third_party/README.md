# Native PNG codec

The native graph executable compiles these C headers directly; no Python image
package is needed for inference or model tensor preparation.

Source: [nothings/stb](https://github.com/nothings/stb/tree/2c980bb59875b0d32144a71867fbdebb2f77cd20),
revision 2c980bb59875b0d32144a71867fbdebb2f77cd20. Each header retains its original
MIT/public-domain license and notices.

| Upstream file | Original SHA256 |
|---|---|
| stb_image.h | 594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3 |
| stb_image_write.h | cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05 |

The engine restricts decoding to PNG, checks framing/CRC and scaled pixel bounds
before decoding, verifies exact inflated scanline size in a fixed buffer, and
bounds encoded output. One local stb_image.h patch rejects pixel indexes outside
the declared palette before reading palette entries; the original function
ignored its palette length. Modified header SHA256:
0cf24a63093060ce3bce350d89a57d08028ff6330705d1cf74eee361688efebc.
SDK-free PNG regressions cover both fixes, all supported color types and Adam7.
The neural result comes from the
trained model; bicubic resizing supplies its second input and the alpha channel.
