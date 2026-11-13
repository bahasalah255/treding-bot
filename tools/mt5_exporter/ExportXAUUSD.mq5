// Read-only historical exporter. It never sends orders.
#property script_show_inputs

input string   SymbolName = "";
input datetime StartDate = D'2024.01.01 00:00';
input datetime EndDate   = D'2025.01.01 00:00';

string EffectiveSymbol()
{
   if(StringLen(SymbolName) > 0)
      return SymbolName;
   return _Symbol;
}

bool ExportTimeframe(const string symbol, const ENUM_TIMEFRAMES timeframe, const string suffix)
{
   MqlRates rates[];
   const int copied = CopyRates(symbol, timeframe, StartDate, EndDate, rates);
   if(copied <= 0)
   {
      PrintFormat("CopyRates failed for %s %s: %d", symbol, suffix, GetLastError());
      return false;
   }

   ArraySetAsSeries(rates, false);
   string file_symbol = symbol;
   StringToLower(file_symbol);
   const string file_name = file_symbol + "_" + suffix + ".csv";
   const int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   const int handle = FileOpen(
      file_name,
      FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_ANSI,
      ','
   );
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("FileOpen failed for %s: %d", file_name, GetLastError());
      return false;
   }

   FileWrite(
      handle,
      "timestamp", "open", "high", "low", "close",
      "tick_volume", "spread", "real_volume"
   );
   for(int index = 0; index < copied; index++)
   {
      FileWrite(
         handle,
         TimeToString(rates[index].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS) + "Z",
         DoubleToString(rates[index].open, digits),
         DoubleToString(rates[index].high, digits),
         DoubleToString(rates[index].low, digits),
         DoubleToString(rates[index].close, digits),
         rates[index].tick_volume,
         rates[index].spread,
         rates[index].real_volume
      );
   }
   FileClose(handle);
   PrintFormat("Exported %d %s bars to common files/%s", copied, suffix, file_name);
   return true;
}

void OnStart()
{
   const string symbol = EffectiveSymbol();
   if(!SymbolSelect(symbol, true))
   {
      PrintFormat("Cannot select symbol %s: %d", symbol, GetLastError());
      return;
   }
   if(StartDate >= EndDate)
   {
      Print("StartDate must be earlier than EndDate");
      return;
   }

   const bool m5_ok = ExportTimeframe(symbol, PERIOD_M5, "M5");
   const bool m15_ok = ExportTimeframe(symbol, PERIOD_M15, "M15");
   if(!m5_ok || !m15_ok)
      Print("Export completed with errors; inspect the Experts log.");
}
