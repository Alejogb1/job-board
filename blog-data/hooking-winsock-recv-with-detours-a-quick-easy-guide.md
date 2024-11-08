---
title: "Hooking Winsock recv() with Detours:  A Quick & Easy Guide"
date: '2024-11-08'
id: 'hooking-winsock-recv-with-detours-a-quick-easy-guide'
---

```c++
#define _CRT_SECURE_NO_DEPRECATE
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <WinSock2.h>
#include <detours.h>
#include <stdio.h>
#pragma comment(lib, "ws2_32.lib")

FILE *pRecvLogFile;

int (WINAPI *pRecv)(SOCKET s, char *buf, int len, int flags) = 
    (int (WINAPI *)(SOCKET, char *, int, int))GetProcAddress(GetModuleHandle("ws2_32.dll"), "recv");

int WINAPI MyRecv(SOCKET s, char* buf, int len, int flags)
{
    int read = pRecv(s, buf, len, flags);
    if (read <= 0)
    {
        return read;
    }

    fopen_s(&pRecvLogFile, "C:\\RecvLog.txt", "a+b");
    fwrite(buf, sizeof(char), read, pRecvLogFile);
    fclose(pRecvLogFile);
    return read;
}

INT APIENTRY DllMain(HMODULE hDLL, DWORD Reason, LPVOID Reserved)
{
    switch (Reason)
    {
        case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(hDLL);

            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            DetourAttach(&(PVOID&)pRecv, MyRecv);
            if (DetourTransactionCommit() == NO_ERROR)
                MessageBox(0, "recv() detoured successfully", "asd", MB_OK);
            break;

        case DLL_PROCESS_DETACH:
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
            break;
    }
    return TRUE;
}
```
