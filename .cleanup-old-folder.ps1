Remove-Item -LiteralPath 'E:\AI_Agent\Harness SDK' -Force -Recurse -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'WindlassCleanupOldFolder' -Confirm:$false -ErrorAction SilentlyContinue
Remove-Item -LiteralPath 'E:\AI_Agent\windlass\.cleanup-old-folder.ps1' -Force -ErrorAction SilentlyContinue
