---
layout:     post
title:      "C# log4net Config Settings"
subtitle:   "C# log4net 设置config"
date:       2018-10-16 20:00:00
author:     "Becks"
header-img: "img/post-bg3.jpg"
catalog:    false
tags:
    - C#
  
---

## Size Rotation log
放在<configuration> 下面, 
```Shell
  <configSections>
    <section name="log4net" type="log4net.Config.Log4NetConfigurationSectionHandler, log4net" />

  </configSections>


  <log4net>
    <appender name="RollingFileAppender" type="log4net.Appender.RollingFileAppender, log4net">
      <file value="C:\log.log" />
      <appendToFile value="true" />
      <rollingStyle value=" Size" />
      <maximumFileSize value="20MB" />
      <maxSizeRollBackups value="50" />
      <staticLogFileName value="true" />
      <layout type="log4net.Layout.PatternLayout">
        <conversionPattern value="%date{yyyy-MM-dd HH:mm:ss.fff} [%level]- {%logger{1}-Line: %L} - %message%newline%exception" />
      </layout>
    </appender>
    <root>
      <level value="DEBUG" />
      <appender-ref ref="RollingFileAppender" />

    </root>

  </log4net>
  ```
  
  ## Time Rotation log
  ```
    <configSections>
    <section name="log4net" type="log4net.Config.Log4NetConfigurationSectionHandler, log4net" />

  </configSections>


  <log4net>
    <appender name="RollingFileAppender" type="log4net.Appender.RollingFileAppender, log4net">
      <lockingModel type="log4net.Appender.FileAppender+MinimalLock"/>
      <file value="C:\Divisa\FIX Trading Maker\pricing acceptor\pricing.log" />
      <datePattern value="yyyy.MM.dd.log" />
      <appendToFile value="true" />
      <PreserveLogFileNameExtension value="true" />
      <rollingStyle value="Date" />
      <maxSizeRollBackups value="50" />
      <layout type="log4net.Layout.PatternLayout">
        <conversionPattern value="%date{yyyy-MM-dd HH:mm:ss.fff} [%level] %message%newline%exception" />
      </layout>
    </appender>
    <root>
      <level value="DEBUG" />
      <appender-ref ref="RollingFileAppender" />

    </root>

  </log4net>
  ```
