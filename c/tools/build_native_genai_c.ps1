<# Rebuild the official pinned SDK C wrapper with declared C API extensions.
   The Colibri client/core remains C. No model or user asset is modified. #>
[CmdletBinding()]
param(
    [string]$SdkRoot,
    [string]$SourceRoot,
    [string]$OutputRoot,
    [switch]$FetchSource,
    [string]$VcVars = 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$discoveryRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'outputs-native/native-sdk-discovery'))
if (-not $SdkRoot) { $SdkRoot = Join-Path $discoveryRoot 'sdk-2026.4.1/openvino_genai_windows_2026.4.1.0_x86_64' }
if (-not $SourceRoot) { $SourceRoot = Join-Path $discoveryRoot 'pinned-c-api-source' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $discoveryRoot 'patched-c-api' }
$SdkRoot = [IO.Path]::GetFullPath($SdkRoot)
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$VcVars = [IO.Path]::GetFullPath($VcVars)
foreach ($value in @($SdkRoot, $SourceRoot, $OutputRoot, $VcVars)) {
    if ($value.IndexOfAny([char[]]@([char]0, [char]10, [char]13, [char]34, [char]37)) -ge 0) {
        throw 'Paths cannot contain NUL, quotes, newlines or batch-variable delimiters'
    }
}
if (-not $OutputRoot.StartsWith($discoveryRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Build output must remain under outputs-native/native-sdk-discovery'
}
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output directory must be new; preserve existing SDK builds' }
if (-not (Test-Path -LiteralPath $VcVars -PathType Leaf)) { throw 'vcvars64.bat is absent' }
$release = Join-Path $SdkRoot 'runtime/bin/intel64/Release'
$imports = Join-Path $SdkRoot 'runtime/lib/intel64/Release'
$include = Join-Path $SdkRoot 'runtime/include'
$versionFile = Join-Path $SdkRoot 'runtime/cmake/OpenVINOGenAIConfigVersion.cmake'
if (-not (Test-Path -LiteralPath $versionFile) -or
    -not ([IO.File]::ReadAllText($versionFile).Contains('set(PACKAGE_VERSION "2026.4.1.0")'))) {
    throw 'Expected official matching GenAI SDK 2026.4.1.0'
}
foreach ($name in @('openvino_genai.lib', 'openvino_c.lib', 'openvino.lib')) {
    if (-not (Test-Path -LiteralPath (Join-Path $imports $name) -PathType Leaf)) { throw "Missing SDK import: $name" }
}
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'native_genai_c_source_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.commit -ne 'ab01d1771e4adaa0e1e08e9d630def170572506c') { throw 'Unexpected source pin' }
function Git-BlobHash([byte[]]$bytes) {
    $prefix = [Text.Encoding]::ASCII.GetBytes('blob ' + $bytes.Length + [char]0)
    $hash = [Security.Cryptography.SHA1]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash([byte[]]($prefix + $bytes)))).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose() }
}
if ($FetchSource -and -not (Test-Path -LiteralPath $SourceRoot)) {
    if (-not $SourceRoot.StartsWith($discoveryRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Fetched source must remain under outputs-native/native-sdk-discovery'
    }
    foreach ($entry in $manifest.sources) {
        if (-not ($entry.path.StartsWith('src/c/include/') -or $entry.path.StartsWith('src/c/src/')) -or
            $entry.path.Contains('..') -or $entry.path.Contains(':') -or $entry.path.Contains('\')) { throw 'Invalid pinned source path' }
        $destination = Join-Path $SourceRoot $entry.path
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
        Invoke-WebRequest -Uri ('https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/' + $manifest.commit + '/' + $entry.path) -OutFile $destination -TimeoutSec 30
    }
}
# Verify every original file before creating or modifying the build tree.
foreach ($entry in $manifest.sources) {
    if (-not ($entry.path.StartsWith('src/c/include/') -or $entry.path.StartsWith('src/c/src/')) -or
        $entry.path.Contains('..') -or $entry.path.Contains(':') -or $entry.path.Contains('\')) { throw 'Invalid pinned source path' }
    $source = Join-Path $SourceRoot $entry.path
    if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or
        (Git-BlobHash ([IO.File]::ReadAllBytes($source))) -ne $entry.sha) { throw "Pinned source mismatch: $($entry.path)" }
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$buildSource = Join-Path $OutputRoot 'source'
foreach ($entry in $manifest.sources) {
    $target = Join-Path $buildSource $entry.path
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $SourceRoot $entry.path) -Destination $target
}
$header = Join-Path $buildSource 'src/c/include/openvino/genai/c/generation_config.h'
$implementation = Join-Path $buildSource 'src/c/src/generation_config.cpp'
$headerMarker = 'OPENVINO_GENAI_C_EXPORTS void ov_genai_generation_config_free(ov_genai_generation_config* handle);'
$functionMarker = 'ov_status_e ov_genai_generation_config_set_max_new_tokens('
$declaration = @'


/* Set whether the SDK applies its chat template to a non-chat string request. */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_apply_chat_template(
    ov_genai_generation_config* config, const bool value);
'@
$function = @'
ov_status_e ov_genai_generation_config_set_apply_chat_template(ov_genai_generation_config* config, const bool value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->apply_chat_template = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

'@
$headerText = [IO.File]::ReadAllText($header)
$implementationText = [IO.File]::ReadAllText($implementation)
if (($headerText.Split([string[]]@($headerMarker), [StringSplitOptions]::None).Length -ne 2) -or
    ($implementationText.Split([string[]]@($functionMarker), [StringSplitOptions]::None).Length -ne 2)) { throw 'Patch anchor mismatch' }
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($header, $headerText.Replace($headerMarker, $headerMarker + $declaration), $utf8)
[IO.File]::WriteAllText($implementation, $implementationText.Replace($functionMarker, $function + $functionMarker), $utf8)
$whisperHeader = Join-Path $buildSource 'src/c/include/openvino/genai/c/whisper_generation_config.h'
$whisperImplementation = Join-Path $buildSource 'src/c/src/whisper_generation_config.cpp'
$llmHeader = Join-Path $buildSource 'src/c/include/openvino/genai/c/llm_pipeline.h'
$llmImplementation = Join-Path $buildSource 'src/c/src/llm_pipeline.cpp'
$whisperDeclaration = @'
/* Set the inherited generation bound on the actual owned Whisper configuration. */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_whisper_generation_config_set_max_new_tokens(
    ov_genai_whisper_generation_config* config, const size_t value);

'@
$whisperFunction = @'
ov_status_e ov_genai_whisper_generation_config_set_max_new_tokens(
    ov_genai_whisper_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->max_new_tokens = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

'@
$renderDeclaration = @'
/* Render the model's public tokenizer template with add_generation_prompt=true.
 * Accepts 1..64 system/user/assistant UTF-8 messages, at most 1 MiB content total.
 * output=NULL queries bytes including NUL. A failed call publishes no output/size.
 * Rendered output is capped at 1 MiB including NUL. No conversation state changes.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_render_chat_template(
    const ov_genai_llm_pipeline* pipe, const char* const* roles, const char* const* contents,
    const size_t count, char* output, size_t* output_size);

'@
$renderFunction = @'
namespace {
bool colibri_bounded_utf8_size(const char* text, const size_t limit, size_t& length) {
    if (!text) return false;
    size_t i = 0;
    while (i <= limit && text[i]) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t width;
        unsigned int value;
        if (lead < 0x80) { width = 1; value = lead; }
        else if (lead >= 0xc2 && lead <= 0xdf) { width = 2; value = lead & 0x1f; }
        else if (lead >= 0xe0 && lead <= 0xef) { width = 3; value = lead & 0x0f; }
        else if (lead >= 0xf0 && lead <= 0xf4) { width = 4; value = lead & 0x07; }
        else return false;
        if (width > limit - i) return false;
        for (size_t j = 1; j < width; ++j) {
            const unsigned char byte = static_cast<unsigned char>(text[i + j]);
            if ((byte & 0xc0) != 0x80) return false;
            value = (value << 6) | (byte & 0x3f);
        }
        if ((width == 3 && value < 0x800) || (width == 4 && value < 0x10000) ||
            value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
        i += width;
    }
    if (i > limit) return false;
    length = i;
    return true;
}
}

ov_status_e ov_genai_llm_pipeline_render_chat_template(
    const ov_genai_llm_pipeline* pipe, const char* const* roles, const char* const* contents,
    const size_t count, char* output, size_t* output_size) {
    if (!pipe || !(pipe->object) || !roles || !contents || !output_size || count == 0 || count > 64) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        const size_t content_limit = 1024 * 1024;
        size_t total = 0;
        ov::genai::ChatHistory history;
        for (size_t i = 0; i < count; ++i) {
            size_t role_size = 0, content_size = 0;
            if (!colibri_bounded_utf8_size(roles[i], 16, role_size) ||
                !colibri_bounded_utf8_size(contents[i], content_limit - total, content_size)) {
                return ov_status_e::INVALID_C_PARAM;
            }
            const std::string role(roles[i], role_size);
            if (role != "system" && role != "user" && role != "assistant") {
                return ov_status_e::INVALID_C_PARAM;
            }
            total += content_size;
            history.push_back(ov::AnyMap{{"role", role}, {"content", std::string(contents[i], content_size)}});
        }
        const std::string rendered = pipe->object->get_tokenizer().apply_chat_template(history, true);
        if (rendered.size() >= content_limit || rendered.find('\0') != std::string::npos) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        const size_t needed = rendered.size() + 1;
        if (output && *output_size < needed) return ov_status_e::OUT_OF_BOUNDS;
        if (output) memcpy(output, rendered.c_str(), needed);
        *output_size = needed;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

'@
function Insert-Before([string]$path, [string]$marker, [string]$insertion) {
    $value = [IO.File]::ReadAllText($path)
    if ($value.Split([string[]]@($marker), [StringSplitOptions]::None).Length -ne 2) { throw "Patch anchor mismatch: $path" }
    [IO.File]::WriteAllText($path, $value.Replace($marker, $insertion + $marker), $utf8)
}
Insert-Before $whisperHeader "OPENVINO_GENAI_C_EXPORTS ov_status_e`nov_genai_whisper_generation_config_get_generation_config(" $whisperDeclaration
Insert-Before $whisperImplementation 'ov_status_e ov_genai_whisper_generation_config_get_generation_config(' $whisperFunction
Insert-Before $llmHeader 'OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_generate(' $renderDeclaration
Insert-Before $llmImplementation 'ov_status_e ov_genai_llm_pipeline_generate(' $renderFunction
foreach ($pipelineSource in @($llmImplementation, (Join-Path $buildSource 'src/c/src/whisper_pipeline.cpp'))) {
    $text = [IO.File]::ReadAllText($pipelineSource)
    $narrowPath = 'std::filesystem::path(models_path)'
    if ($text.Split([string[]]@($narrowPath), [StringSplitOptions]::None).Length -ne 2) { throw 'Pipeline path anchor mismatch' }
    [IO.File]::WriteAllText($pipelineSource, $text.Replace($narrowPath, 'std::filesystem::u8path(models_path)'), $utf8)
}
$cpp = @($manifest.sources | Where-Object { $_.path.EndsWith('.cpp') } | ForEach-Object { '"' + (Join-Path $buildSource $_.path) + '"' })
$compile = 'cl /nologo /LD /EHsc /std:c++17 /MD /O2 /DNDEBUG /Dopenvino_genai_c_EXPORTS /I"' +
    (Join-Path $buildSource 'src/c/include') + '" /I"' + $include + '" ' + ($cpp -join ' ') +
    ' /link /OUT:openvino_genai_c.dll /IMPLIB:openvino_genai_c.lib /LIBPATH:"' + $imports +
    '" openvino_genai.lib openvino_c.lib openvino.lib /Brepro'
$batch = "@echo off`r`nsetlocal DisableDelayedExpansion`r`ncall `"$VcVars`" > nul`r`nif errorlevel 1 exit /b %errorlevel%`r`n$compile`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText((Join-Path $OutputRoot 'build_vendor.cmd'), $batch, [Text.Encoding]::ASCII)
Push-Location $OutputRoot
try { & $env:ComSpec /d /c build_vendor.cmd; if ($LASTEXITCODE -ne 0) { throw "SDK C wrapper compilation failed: $LASTEXITCODE" } }
finally { Pop-Location }
$dll = Join-Path $OutputRoot 'openvino_genai_c.dll'
$record = @{sdk_version='2026.4.1.0';source_commit=$manifest.commit;source_root=$SourceRoot;sdk_root=$SdkRoot;
    added_exports=@('ov_genai_generation_config_set_apply_chat_template',
        'ov_genai_whisper_generation_config_set_max_new_tokens',
        'ov_genai_llm_pipeline_render_chat_template');dll=$dll;
    dll_sha256=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant();
    patch_sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'native_genai_c_apply_template.patch') -Algorithm SHA256).Hash.ToLowerInvariant();
    client_language='C';vendor_wrapper_language='official C++ implementation behind C ABI'}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputRoot 'build-manifest.json') -Encoding utf8
$record | ConvertTo-Json -Depth 4
